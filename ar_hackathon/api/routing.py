"""
Amazon Robotics Hackathon - Routing API

Score-guided joint planning with automatic batching and congestion-aware routes.

*****IMPORTANT*****
Team name: Robot Warriors
Email addresses:
    larryw01@student.ubc.ca
    cli113@student.ubc.ca
    losandhu@student.ubc.ca
    hsing101@student.ubc.ca
*******************
"""

from heapq import heappop, heappush, nlargest
from math import ceil, exp
from time import perf_counter
from typing import Optional
from ar_hackathon.models.graph_state import GraphState


_planner = None
_last_call = (-1, -1)
_INF = 10**9
_count = getattr(int, "bit_count", lambda mask: bin(mask).count("1"))


class _SearchExpired(Exception):
    pass


class _Distances:
    """Cache shortest paths only from locations the search actually uses."""

    def __init__(self, adjacency, node_cap, edge_cap):
        self.adj = adjacency
        self.node_cap = node_cap
        self.edge_cap = edge_cap
        self.cache = {}
        self.next_hops = {}

    def __getitem__(self, src):
        if src in self.cache:
            return self.cache[src]
        distances = [_INF] * len(self.adj)
        next_hops = [None] * len(self.adj)
        distances[src] = 0
        queue = [(0, src)]
        while queue:
            cost, node = heappop(queue)
            if cost != distances[node]:
                continue
            for dst, weight, eid in self.adj[node]:
                if self.edge_cap[eid] == 0 or self.node_cap[dst] == 0:
                    continue
                candidate = cost + weight
                if candidate < distances[dst]:
                    distances[dst] = candidate
                    next_hops[dst] = dst if node == src else next_hops[node]
                    heappush(queue, (candidate, dst))
        self.cache[src] = distances
        self.next_hops[src] = next_hops
        return distances

    def next_hop(self, src, dst):
        self[src]
        return self.next_hops[src][dst]


class _Planner:
    """Replan joint moves as tasks arrive, using the engine's exact rules."""

    def __init__(self, state):
        self.ids = [n.id for n in state.nodes]
        self.index = {node: i for i, node in enumerate(self.ids)}
        self.capacity = [n.capacity for n in state.nodes]
        self.storage = [
            i for i, n in enumerate(state.nodes) if n.node_type == "storage"
        ]
        self.adj = [[] for _ in self.ids]
        self.edge_cap = [e.capacity for e in state.edges]
        self.edge_lookup = {}
        for eid, edge in enumerate(state.edges):
            a, b = self.index[edge.from_node], self.index[edge.to_node]
            pairs = [(a, b), (b, a)] if edge.bidirectional else [(a, b)]
            for src, dst in pairs:
                if (src, dst) not in self.edge_lookup:
                    self.edge_lookup[src, dst] = eid
                    self.adj[src].append((dst, max(1, ceil(edge.weight)), eid))
        self.edge_conflicts = []
        for edge in state.edges:
            a, b = self.index[edge.from_node], self.index[edge.to_node]
            pairs = [(a, b), (b, a)] if edge.bidirectional else [(a, b)]
            self.edge_conflicts.append(
                {(self.edge_lookup[src, dst], dst) for src, dst in pairs}
            )
        self.dist = _Distances(self.adj, self.capacity, self.edge_cap)
        units = sorted(state.drive_units, key=lambda u: u.id)
        self.unit_ids = [u.id for u in units]
        self.unit_cap = [u.capacity for u in units]
        self.plan = []
        self.expected = []
        self.known = set()
        self.plan_time = -1
        self.tick = -1
        self.actions = {}
        self.search_seconds = 0.0
        self.observed_pods = set()
        self.arrivals = {}

    def _observe_arrivals(self, state):
        couriers = {u.id: u for u in state.drive_units}
        for pod in state.active_pods:
            if pod.id in self.observed_pods:
                continue
            self.observed_pods.add(pod.id)
            source = pod.current_node
            if source is None and pod.entry_time == state.current_time_step:
                unit = couriers.get(pod.carried_by)
                if unit is not None and not unit.in_transit:
                    source = unit.current_node
            if source is not None:
                source = self.index[source]
                waves = self.arrivals.setdefault(source, {})
                waves.setdefault(pod.entry_time, []).append(
                    self.index[pod.destination_station]
                )

    def _batch_waits(self, units, waiting, now):
        """Wait briefly when a recurring arrival can save another round trip."""
        held = set()
        if waiting:
            return held
        claimed = set()
        for i, (node, rem, _, load) in enumerate(units):
            if rem or not load or _count(load) >= self.unit_cap[i]:
                continue
            if self.capacity[node] is not None or node in claimed:
                continue
            waves = self.arrivals.get(node, {})
            times = sorted(waves)
            if len(times) < 2:
                continue
            recent = times[-4:]
            gaps = [b - a for a, b in zip(recent, recent[1:])]
            period = times[-1] - times[-2]
            if len(times) > 2 and max(gaps) - min(gaps) > 1:
                continue
            delay = times[-1] + period - now
            if not 0 < delay <= min(6, period / 2):
                continue
            gain, duration, end, _ = self._tour(node, load)
            available = duration + self.dist[end][node]
            for j, (other, travel, _, cargo) in enumerate(units):
                if j == i:
                    continue
                _, finish, last, _ = self._tour(other, cargo)
                available = min(available, travel + finish + self.dist[last][node])
            benefit = _INF
            destinations = {
                dst
                for history in self.arrivals.values()
                for wave in history.values()
                for dst in wave
                if self.dist[node][dst] < _INF
            }
            for dst in destinations:
                direct = self.dist[node][dst]
                later = exp(-(max(delay, available) + direct - delay) / 50)
                first = exp(-direct / 50) * (
                    1 + exp(-delay / 50) * self._tour(dst, load)[0]
                )
                last = gain * exp(-delay / 50) + exp(
                    -(duration + self.dist[end][dst]) / 50
                )
                benefit = min(benefit, max(first, last) - gain - later)
            loss = gain * (1 - exp(-delay / 50))
            # Charge for the possibility that the next wave never arrives.
            if destinations and 0.6 * benefit - 0.4 * loss > 0.02:
                held.add(i)
                claimed.add(node)
        return held

    def _read(self, state):
        pods = sorted(state.active_pods, key=lambda p: (p.entry_time, p.id))
        self.pod_index = {p.id: i for i, p in enumerate(pods)}
        self.sources = [
            self.index[p.current_node] if p.current_node is not None else -1
            for p in pods
        ]
        self.dest = [self.index[p.destination_station] for p in pods]
        self.value = [exp((p.entry_time - state.current_time_step) / 50) for p in pods]
        self.at_source = [0] * len(self.ids)
        self.at_dest = [0] * len(self.ids)
        couriers = {u.id: u for u in state.drive_units}
        for i, pod in enumerate(pods):
            self.at_dest[self.dest[i]] |= 1 << i
            if pod.carried_by is None and pod.current_node is not None:
                self.at_source[self.sources[i]] |= 1 << i
                if self.sources[i] not in self.storage:
                    self.storage.append(self.sources[i])
            elif pod.entry_time == state.current_time_step:
                unit = couriers.get(pod.carried_by)
                if unit is not None and not unit.in_transit:
                    source = self.index[unit.current_node]
                    if source not in self.storage:
                        self.storage.append(source)
        self.mask_value = {0: 0.0}
        self.tour_cache = {}
        self.estimate_cache = {}
        self.rank_cache = {}
        self.storage_distances = {}
        self.evaluations = 0
        return self._encode(state)

    def _encode(self, state):
        """Units store reserved node, travel time, edge index, and a pod bitmask."""
        waiting = sum(
            1 << self.pod_index[p.id]
            for p in state.active_pods
            if p.carried_by is None and p.current_node is not None
        )
        units = []
        for unit in sorted(state.drive_units, key=lambda u: u.id):
            load = sum(
                1 << self.pod_index[p] for p in unit.carrying if p in self.pod_index
            )
            if unit.in_transit:
                src, dst = (
                    self.index[unit.current_node],
                    self.index[unit.transit_destination],
                )
                units.append(
                    (
                        dst,
                        max(1, ceil(unit.transit_remaining_time)),
                        self.edge_lookup[src, dst],
                        load,
                    )
                )
            else:
                units.append((self.index[unit.current_node], 0, -1, load))
        return tuple(units), waiting

    def _value(self, mask):
        if mask not in self.mask_value:
            remaining, total = mask, 0.0
            while remaining:
                bit = remaining & -remaining
                remaining ^= bit
                total += self.value[bit.bit_length() - 1]
            self.mask_value[mask] = total
        return self.mask_value[mask]

    def _resolve(self, units, waiting):
        result = list(units)
        delivered = 0
        for i, (node, remaining, eid, load) in enumerate(units):
            if remaining:
                continue
            dropped = load & self.at_dest[node]
            delivered |= dropped
            load ^= dropped
            available = waiting & self.at_source[node]
            room = self.unit_cap[i] - _count(load)
            while available and room > 0:
                bit = available & -available
                available ^= bit
                waiting ^= bit
                load |= bit
                room -= 1
            result[i] = (node, 0, -1, load)
        return tuple(result), waiting, self._value(delivered)

    def _advance(self, units, waiting):
        advanced = tuple(
            (n, max(0, r - 1), e if r > 1 else -1, load) for n, r, e, load in units
        )
        return self._resolve(advanced, waiting)

    def _moves(self, units, i):
        node, remaining, _, load = units[i]
        if remaining:
            return []
        result = []
        for dst, weight, eid in self.adj[node]:
            if dst == node:
                continue
            cap = self.capacity[dst]
            if cap is not None and sum(u[0] == dst for u in units) >= cap:
                continue
            cap = self.edge_cap[eid]
            if (
                cap is not None
                and sum(
                    u[1] > 0 and (u[2], u[0]) in self.edge_conflicts[eid] for u in units
                )
                >= cap
            ):
                continue
            result.append((dst, weight, eid, load))
        return result

    def _tour(self, node, load):
        """Order a small batch's deliveries by its actual discounted reward."""
        key = (node, load)
        if key in self.tour_cache:
            return self.tour_cache[key]
        if not load:
            return (0.0, 0, node, node)
        destinations = []
        mask = load
        while mask:
            bit = mask & -mask
            dst = self.dest[bit.bit_length() - 1]
            destinations.append(dst)
            mask &= ~self.at_dest[dst]
        if len(destinations) > 5:
            current, duration, gain, first = node, 0, 0.0, node
            while destinations:
                dst = min(destinations, key=lambda d: self.dist[current][d])
                distance = self.dist[current][dst]
                if distance >= _INF:
                    break
                if current == node and duration == 0:
                    first = dst
                duration += distance
                gain += exp(-duration / 50) * self._value(load & self.at_dest[dst])
                current = dst
                destinations.remove(dst)
            result = (gain, duration, current, first)
            self.tour_cache[key] = result
            return result
        best = (-1.0, _INF, node, node)
        for dst in destinations:
            distance = self.dist[node][dst]
            if distance >= _INF:
                continue
            delivered = load & self.at_dest[dst]
            gain, duration, end, _ = self._tour(dst, load ^ delivered)
            gain = exp(-distance / 50) * (self._value(delivered) + gain)
            candidate = (gain, distance + duration, end, dst)
            if gain > best[0] + 1e-12 or (
                abs(gain - best[0]) <= 1e-12 and candidate[1] < best[1]
            ):
                best = candidate
        if best[0] < 0:
            best = (0.0, 0, node, node)
        self.tour_cache[key] = best
        return best

    def _estimate(self, units, waiting):
        """Schedule batches once, rather than assigning every pod the same robot."""
        self.evaluations += 1
        if self.evaluations % 32 == 0 and perf_counter() >= self.deadline:
            raise _SearchExpired
        key = (units, waiting)
        cached = self.estimate_cache.get(key)
        if cached is not None:
            return cached
        score = self._schedule_estimate(units, waiting)
        self.estimate_cache[key] = score
        return score

    def _schedule_estimate(self, units, waiting):
        score = 0.0
        workers = []
        for node, rem, _, load in units:
            gain, duration, end, _ = self._tour(node, load)
            score += gain * exp((1 - rem) / 50)
            workers.append((end, rem + duration))
        queues = {
            src: waiting & mask
            for src, mask in enumerate(self.at_source)
            if waiting & mask
        }
        while queues:
            best = None
            for i, (node, available) in enumerate(workers):
                for src, queue in queues.items():
                    distance = self.dist[node][src]
                    if distance >= _INF:
                        continue
                    batch = 0
                    for _ in range(min(_count(queue), self.unit_cap[i])):
                        bit = queue & -queue
                        queue ^= bit
                        batch |= bit
                    if not batch:
                        continue
                    gain, duration, end, _ = self._tour(src, batch)
                    finish = available + distance + duration
                    gain *= exp((1 - available - distance) / 50)
                    priority = gain / (4 + finish)
                    if best is None or priority > best[0]:
                        best = (priority, gain, i, src, batch, finish, end)
            if best is None:
                break
            _, gain, i, src, batch, finish, end = best
            score += gain
            queues[src] ^= batch
            if not queues[src]:
                del queues[src]
            workers[i] = (end, finish)
        return score

    def _targets(self, units, waiting):
        targets = {}
        free = []
        for i, (node, rem, _, load) in enumerate(units):
            if load:
                targets[i] = self._tour(node, load)[3]
            else:
                free.append(i)
        queues = {
            src: waiting & mask
            for src, mask in enumerate(self.at_source)
            if waiting & mask
        }
        while free and queues:
            best = None
            for i in free:
                node, rem, _, _ = units[i]
                for src, queue in queues.items():
                    distance = rem + self.dist[node][src]
                    if distance >= _INF:
                        continue
                    batch = 0
                    for _ in range(min(_count(queue), self.unit_cap[i])):
                        bit = queue & -queue
                        queue ^= bit
                        batch |= bit
                    gain, duration, _, _ = self._tour(src, batch)
                    priority = gain * exp(-distance / 50) / (4 + distance + duration)
                    if best is None or priority > best[0]:
                        best = (priority, i, src, batch)
            if best is None:
                break
            _, i, src, batch = best
            targets[i] = src
            free.remove(i)
            queues[src] ^= batch
            if not queues[src]:
                del queues[src]
        return targets

    def _route(self, units, i, target):
        """Compare detours with the time needed for occupied aisles to clear."""
        start = units[i][0]
        if target == start:
            return None
        queue = [(0, start, -1)]
        costs = {start: 0}
        while queue:
            cost, node, first = heappop(queue)
            if cost != costs[node]:
                continue
            if node == target:
                return None if first < 0 else first
            for dst, weight, eid in self.adj[node]:
                if dst == node:
                    continue
                depart = cost
                cap = self.edge_cap[eid]
                if cap is not None:
                    release = sorted(
                        u[1]
                        for j, u in enumerate(units)
                        if j != i and u[1] and (u[2], u[0]) in self.edge_conflicts[eid]
                    )
                    if len(release) >= cap:
                        depart = max(depart, release[len(release) - cap])
                cap = self.capacity[dst]
                if cap is not None:
                    release = sorted(
                        max(1, u[1] + 1)
                        for j, u in enumerate(units)
                        if j != i and u[0] == dst
                    )
                    if len(release) >= cap:
                        depart = max(depart, release[len(release) - cap])
                arrival = depart + weight
                if arrival < costs.get(dst, _INF):
                    costs[dst] = arrival
                    action = dst if node == start and depart == 0 else first
                    heappush(queue, (arrival, dst, action))
        return None

    def _greedy(self, units, waiting):
        targets = self._targets(units, waiting)
        parking = self._parking(units)
        actions = [None] * len(units)
        current = list(units)
        for i, unit in enumerate(units):
            if unit[1]:
                continue
            target = targets.get(i)
            move = (
                parking[i] if target is None else self._route(tuple(current), i, target)
            )
            if move is None:
                continue
            chosen = next(
                (m for m in self._moves(tuple(current), i) if m[0] == move), None
            )
            if chosen is not None:
                actions[i] = move
                current[i] = chosen
        return tuple(actions), tuple(current)

    def _parking(self, units):
        """Return empty robots to storage while keeping docks available."""
        actions = [None] * len(units)
        current = list(units)
        assigned = {}
        for node, rem, _, load in units:
            if not load and node in self.storage:
                assigned[node] = assigned.get(node, 0) + 1
        for i, unit in enumerate(units):
            node, rem, _, load = unit
            if rem or load:
                continue
            if node in assigned:
                assigned[node] -= 1
            candidates = self.storage or [
                j for j, cap in enumerate(self.capacity) if cap is None
            ]
            if not candidates:
                continue
            target = min(
                candidates, key=lambda s: self.dist[node][s] + 5 * assigned.get(s, 0)
            )
            if node == target:
                assigned[target] = assigned.get(target, 0) + 1
                continue
            choices = self._moves(tuple(current), i)
            if choices:
                move = min(choices, key=lambda m: m[1] + self.dist[m[0]][target])
                if move[1] + self.dist[move[0]][target] < _INF:
                    current[i] = move
                    actions[i] = move[0]
                    assigned[target] = assigned.get(target, 0) + 1
        return tuple(actions)

    def _idle_cost(self, units):
        if not self.storage:
            return 0
        cost = 0
        for node, rem, _, load in units:
            if load:
                continue
            if node not in self.storage_distances:
                self.storage_distances[node] = min(
                    self.dist[node][src] for src in self.storage
                )
            cost += rem + self.storage_distances[node]
        return cost

    def _step_rank(self, units, waiting):
        """Reuse scores when different search branches reach the same state."""
        key = (units, waiting)
        cached = self.rank_cache.get(key)
        if cached is not None:
            return cached
        units, waiting, gained = self._advance(units, waiting)
        result = (
            gained + exp(-1 / 50) * self._estimate(units, waiting),
            -self._idle_cost(units),
        )
        self.rank_cache[key] = result
        return result

    def _upper_bound(self, units, waiting):
        score = 0.0
        for node, rem, _, load in units:
            while load:
                bit = load & -load
                load ^= bit
                p = bit.bit_length() - 1
                delay = max(0, rem + self.dist[node][self.dest[p]] - 1)
                score += self.value[p] * exp(-delay / 50)
        while waiting:
            bit = waiting & -waiting
            waiting ^= bit
            p = bit.bit_length() - 1
            src = self.sources[p]
            best = _INF
            for i, (node, rem, _, load) in enumerate(units):
                pickup = self.dist[node][src]
                if _count(load) >= self.unit_cap[i]:
                    pickup = _INF
                    while load:
                        bit = load & -load
                        load ^= bit
                        dst = self.dest[bit.bit_length() - 1]
                        pickup = min(pickup, self.dist[node][dst] + self.dist[dst][src])
                best = min(best, rem + pickup + self.dist[src][self.dest[p]])
            score += self.value[p] * exp(-max(0, best - 1) / 50)
        return score

    def _search(self, units, waiting, budget=0.35):
        started = perf_counter()
        budget = min(budget, max(0, 40 - self.search_seconds))
        self.deadline = started + budget
        actions, _ = self._greedy(units, waiting)
        self.incumbent = (actions,)
        if budget == 0:
            return self.incumbent
        try:
            return self._beam_search(units, waiting)
        except _SearchExpired:
            return self.incumbent
        finally:
            self.search_seconds += perf_counter() - started

    def _beam_search(self, units, waiting):
        deadline = self.deadline
        width = max(32, min(256, 768 // max(1, len(units))))
        # Each layer follows the engine's ascending unit order.
        beam = [(units, waiting, 0.0, ())]
        best_path = self.incumbent
        best_complete = -1.0
        complete_path = ()
        us, w, reward, path = units, waiting, 0.0, ()
        seed_deadline = min(deadline, perf_counter() + 0.025)
        for depth in range(64):
            if perf_counter() >= seed_deadline:
                break
            if depth:
                us, w, gained = self._resolve(us, w)
                reward += exp(-depth / 50) * gained
            actions, committed = self._greedy(us, w)
            us, w, gained = self._advance(committed, w)
            reward += exp(-depth / 50) * gained
            path += (actions,)
            if not w and not any(u[3] for u in us):
                best_path, best_complete = path, reward
                complete_path = path
                self.incumbent = best_path
                break
        if best_complete >= self._upper_bound(units, waiting) - 1e-10:
            return best_path
        discount = 1.0
        for depth in range(48):
            if depth:
                normalized = []
                for us, w, reward, path in beam:
                    us, w, gained = self._resolve(us, w)
                    normalized.append((us, w, reward + discount * gained, path))
                beam = normalized
            expanded = [(us, w, reward, path, ()) for us, w, reward, path in beam]
            for i in range(len(units)):
                if all(us[i][1] for us, _, _, _, _ in expanded):
                    expanded = [
                        (us, w, reward, path, actions + (None,))
                        for us, w, reward, path, actions in expanded
                    ]
                    continue
                candidates = {}
                for us, w, reward, path, actions in expanded:
                    options = [None] + self._moves(us, i)
                    for move in options:
                        next_units = us
                        if move is not None:
                            next_units = us[:i] + (move,) + us[i + 1 :]
                        acts = actions + (None if move is None else move[0],)
                        key = (next_units, w)
                        prior = candidates.get(key)
                        if prior is None or reward > prior[2]:
                            candidates[key] = (next_units, w, reward, path, acts)

                def partial_rank(item):
                    us, w, reward, _, _ = item
                    value, idle = self._step_rank(us, w)
                    return (reward + discount * value, idle)

                expanded = nlargest(width * 2, candidates.values(), key=partial_rank)
                if perf_counter() >= deadline:
                    return best_path
            candidates = {}
            for us, w, reward, path, actions in expanded:
                us, w, gained = self._advance(us, w)
                reward += discount * gained
                path = path + (actions,)
                if not w and not any(u[3] for u in us):
                    if reward > best_complete:
                        best_complete = reward
                        best_path = path
                        complete_path = path
                        self.incumbent = best_path
                    continue
                key = (us, w)
                prior = candidates.get(key)
                if prior is None or reward > prior[2]:
                    candidates[key] = (us, w, reward, path)
            discount *= exp(-1 / 50)
            ranked = nlargest(
                width,
                candidates.values(),
                key=lambda x: (
                    x[2] + discount * self._estimate(x[0], x[1]),
                    -self._idle_cost(x[0]),
                ),
            )
            if not ranked:
                if complete_path:
                    best_path = complete_path
                break
            forecast = ranked[0][2] + discount * self._estimate(ranked[0][0], ranked[0][1])
            best_path = ranked[0][3] if forecast > best_complete + 1e-10 else complete_path
            self.incumbent = best_path
            beam = ranked
            if (
                best_complete
                >= max(x[2] + discount * self._upper_bound(x[0], x[1]) for x in beam)
                - 1e-10
            ):
                best_path = complete_path
                break
            if perf_counter() >= deadline:
                break
        return best_path

    def next(self, uid, state):
        if len(self.unit_ids) == 1 and self.unit_cap[0] == 1:
            unit = state.drive_units[0]
            sources = {
                p.current_node
                for p in state.active_pods
                if p.carried_by is None and p.current_node is not None
            }
            # A single robot needs no joint search for a forced delivery or pickup.
            if unit.carrying or len(sources) <= 1:
                units, waiting = self._read(state)
                node, remaining, _, load = units[0]
                if remaining:
                    return None
                if load:
                    target = self.dest[(load & -load).bit_length() - 1]
                elif waiting:
                    target = self.sources[(waiting & -waiting).bit_length() - 1]
                elif self.storage:
                    target = min(self.storage, key=self.dist[node].__getitem__)
                else:
                    target = node
                self.plan = []
                move = self.dist.next_hop(node, target)
                return None if move is None else self.ids[move]

        now = state.current_time_step
        if now != self.tick:
            self.tick = now
            self._observe_arrivals(state)
            pods = {p.id for p in state.active_pods}
            offset = now - self.plan_time
            batching = any(
                not u.in_transit
                and 0 < len(u.carrying) < u.capacity
                and self.index[u.current_node] in self.arrivals
                for u in state.drive_units
            )
            fresh = (
                batching
                or pods - self.known
                or not (0 <= offset < len(self.plan))
                or offset >= 4
            )
            if not fresh:
                fresh = self._encode(state) != self.expected[offset]
            if fresh:
                units, waiting = self._read(state)
                self.plan = (
                    self._search(units, waiting) if pods else [self._parking(units)]
                )
                self.plan_time = now
                self.known = pods
                offset = 0
                if not self.plan:
                    self.plan = [self._parking(units)]
                self.expected = []
                us, w = units, waiting
                for actions in self.plan:
                    self.expected.append((us, w))
                    current = list(us)
                    for i, target in enumerate(actions):
                        if target is not None:
                            move = next(
                                (
                                    m
                                    for m in self._moves(tuple(current), i)
                                    if m[0] == target
                                ),
                                None,
                            )
                            if move is not None:
                                current[i] = move
                    us, w, _ = self._advance(tuple(current), w)
                    us, w, _ = self._resolve(us, w)
            actions = self.plan[offset]
            self.actions = dict(zip(self.unit_ids, actions))
            if batching:
                held = self._batch_waits(units, waiting, now)
                for i in held:
                    self.actions[self.unit_ids[i]] = None
                if held:
                    self.plan = []
        move = self.actions.get(uid)
        if move is None:
            return None
        unit = state.get_drive_unit(uid)
        target = self.ids[move]
        edge = state.get_edge(unit.current_node, target)
        node = state.get_node(target)
        if (
            unit.in_transit
            or edge is None
            or target == unit.current_node
            or (
                edge.capacity is not None
                and state.edge_occupancy(unit.current_node, target) >= edge.capacity
            )
            or (
                node.capacity is not None
                and state.node_occupancy(target) >= node.capacity
            )
        ):
            self.plan = []
            return None
        return target


def drive_unit_next_move(drive_unit_id: int, state: GraphState) -> Optional[int]:
    """Choose a legal move using only the currently visible warehouse state."""
    global _planner, _last_call
    call = (state.current_time_step, drive_unit_id)
    signature = (
        tuple((n.id, n.node_type, n.capacity) for n in state.nodes),
        tuple(
            (e.from_node, e.to_node, e.weight, e.capacity, e.bidirectional)
            for e in state.edges
        ),
        tuple(sorted((u.id, u.capacity) for u in state.drive_units)),
    )
    if _planner is None or call <= _last_call or signature != _planner.signature:
        _planner = _Planner(state)
        _planner.signature = signature
    _last_call = call
    return _planner.next(drive_unit_id, state)

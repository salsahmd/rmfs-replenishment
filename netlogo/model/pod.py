from engine.object import Object
from engine.netlogo_coordinate import NetLogoCoordinate


class Pod(Object):
    def __init__(self, pod_id: int):
        self.pod_id = pod_id
        self.shape = 'full square'
        self.object_type = 'pod'
        self.coordinate = NetLogoCoordinate()
        self.skus = {}
        self.is_idle = True
        self.station = None
        self.need_replenishment = False
        self.mass = 0
        super().__init__()

    def __eq__(self, other):
        if isinstance(other, Pod):
            return self.pod_id == other.pod_id
        return False

    def __hash__(self):
        return hash(self.pod_id)

    def __repr__(self):
        return f"Pod({self.pod_id})"

    def add_sku(self, sku, slot_id, limit_qty, current_qty, threshold, weight, rop_per_slot=None):
        """Add a slot for a SKU. Same SKU can have multiple slots — each stored separately."""
        if sku not in self.skus:
            self.skus[sku] = {'slots': [], 'threshold': threshold, 'weight': weight}
        self.skus[sku]['slots'].append({
            'slot_id': slot_id,
            'limit_qty': limit_qty,
            'current_qty': current_qty,
            'rop_per_slot': rop_per_slot,
        })
        self.mass += weight * current_qty

    def check_replenishment_needed(self, kl_alpha=0.5):
        """KL gate: pod dispatched when >= kl_alpha fraction of total slots are flagged.

        A slot is flagged when current_qty <= rop_per_slot.
        Fallback (rop_per_slot=0 or None): slot flagged when current_qty / limit_qty <= threshold.
        kl_alpha configurable: 0.3 (aggressive), 0.5 (moderate), 0.7 (conservative).
        """
        flagged_slots = 0
        total_slots = 0
        for sku, data in self.skus.items():
            for slot in data['slots']:
                total_slots += 1
                rop = slot.get('rop_per_slot')
                if rop is not None and rop > 0:
                    if float(slot['current_qty']) <= float(rop):
                        flagged_slots += 1
                else:
                    # fallback for rop=0 slots: flag when slot is at or below threshold ratio
                    if slot['limit_qty'] > 0 and slot['current_qty'] / slot['limit_qty'] <= float(data['threshold']):
                        flagged_slots += 1
        if total_slots == 0:
            return False
        return flagged_slots >= total_slots * kl_alpha

    def replenish_all_skus(self):
        """Replenish all SKUs by resetting each slot's current_qty to its limit_qty."""
        self.mass = 0
        for sku, data in self.skus.items():
            for slot in data['slots']:
                slot['current_qty'] = slot['limit_qty']
                self.mass += data['weight'] * slot['limit_qty']

    def pick_sku(self, sku, qty):
        """Deduct qty from this SKU's slots, taking from highest-qty slot first (FIFO)."""
        remaining = qty
        for slot in sorted(self.skus[sku]['slots'], key=lambda s: -s['current_qty']):
            take = min(remaining, slot['current_qty'])
            slot['current_qty'] -= take
            remaining -= take
            if remaining == 0:
                break
        self.mass -= self.skus[sku]['weight'] * qty

    def get_quantity(self, sku):
        """Return total current qty for a SKU across all its slots in this pod."""
        return sum(s['current_qty'] for s in self.skus[sku]['slots'])

    def get_unassigned_skus(self):
        """Return a list of SKUs that have not yet been assigned a pod."""
        unassigned_skus = [sku for sku, details in self.skus.items() if details.get('pod') is None]
        return unassigned_skus

    def set_pod_station(self, station):
        self.station = station
        return

    def remove_pod_station(self):
        self.station = None
        return

    def get_skus_in_pod(self):
        return self.skus

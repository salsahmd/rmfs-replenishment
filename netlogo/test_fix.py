"""Quick test to verify the _objects fix."""
import pickle
from model.inventory import Inventory

# Create two separate Inventory instances
u1 = Inventory()
u2 = Inventory()

# Verify they have SEPARATE _objects lists
print("u1._objects is u2._objects:", u1._objects is u2._objects)

# Add a dummy object to u1
class FakeObj:
    object_type = "test"
    _id = 0
    heading = 0
    shape = ""
    velocity = 0
    acceleration = 0
    pos_x = 0
    pos_y = 0
    color = 0
    id = 0
    def setUniverse(self, u): pass

u1.addObject(FakeObj())
print("u1 objects:", len(u1._objects))
print("u2 objects:", len(u2._objects), " (should be 0)")

# Check instance attrs are properly set
print()
print("u1 has instance _objects:", "_objects" in u1.__dict__)
print("u1 has instance stop_and_go:", "stop_and_go" in u1.__dict__)
print("u1 has instance total_energy:", "total_energy" in u1.__dict__)
print("u1 has instance movement_channel:", "movement_channel" in u1.__dict__)

# Verify pickle roundtrip preserves _objects
data = pickle.dumps(u1)
u1_loaded = pickle.loads(data)
print()
print("Loaded u1 objects:", len(u1_loaded._objects), " (should be 1)")
print("Loaded has instance _objects:", "_objects" in u1_loaded.__dict__)
print()
print("ALL CHECKS PASSED" if (
    not (u1._objects is u2._objects) and
    len(u1._objects) == 1 and
    len(u2._objects) == 0 and
    "_objects" in u1.__dict__ and
    "_objects" in u1_loaded.__dict__ and
    len(u1_loaded._objects) == 1
) else "SOME CHECKS FAILED")

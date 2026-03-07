class Universe:
    _tick = 0
    id = 0
    tick_to_second = 0.5
    landscape = None
    graph = None
    graph_pod = None
    deadlock_prevention_manager = None

    def __init__(self):
        # These MUST be instance attributes — class-level mutable defaults
        # are shared across all instances and are NOT included in pickle state.
        self._objects = []
        self.warehouse_size = []

    def addObject(self, object):
        object.id = len(self._objects)
        object.setUniverse(self)
        self._objects.append(object)
    
    def set_warehouse_size(self, size):
        self.warehouse_size = size

    def get_warehouse_size(self):
        return self.warehouse_size

    def tick(self):
        for o in self._objects:
            o.move()

    def get_movable_objects(self):
        return self._objects
    
    def generateResult(self):
        result = []
        for o in self.get_movable_objects():
            result.append({
                'id': o.id,
                'heading': o.heading,
                'shape': o.shape,
                'velocity': o.velocity,
                'acceleration': o.acceleration,
                'pos_x': o.pos_x,
                'pos_y': o.pos_y,
                'color': o.color,
            })

        return result


"""
Object to Object relationship models
"""
import numpy as np
from scipy.spatial.distance import euclidean
import ipdb
from collections import OrderedDict

# Params
OBJ_W_SIZE = 0.02  # Object width size
NEAR_DIST_TH = 0.1  # Near distance threshold
HOLD_DIST_TH = 0.02 # Holding distance threshold
IN_DIST_TH = 0.01 # Object in target distance threshold

class SpatialObject(object):

    OBJ_CLASSES = OrderedDict([
        ("eef", 0),
        ("can", 1),
        ("milk", 2),
        ("bread", 3),
        ("cereal", 4),
        ("roundnut", 5),
        ("squarenut", 6)
    ])

    def __init__(self, name, pos, orient, is_robot=False):
        """ Constructor

            Args
            -----
            name: str
                Object name
            pos: np.ndarray
                Position vector (3,1)
            orient: np.ndarray 
                Orientation Quaternion  (4,1)
        """
        self.name = name
        self.pos = pos
        self.orient = orient
        self.is_robot = is_robot
        self.encode_onehot()

    def encode_onehot(self):
        """ Encode into One-hot vector
        """
        self.encoding = np.array([ 
                1. if self.name == _c else 0. \
                    for _c in self.OBJ_CLASSES 
            ])

    def __repr__(self):
        return "<" + self.name + ">"


class Relation(object):

    def __init__(self, name, validator):
        """ Constructor
        
            Args
            -----
            name : str
                Name of the relation
            validator : function | lambda
                Validation function
        """
        self._name = name
        self._f_validator = validator

    @property
    def name(self):
        """ Name getter
        """
        return self._name

    @property
    def validator(self):
        """ Validator Function getter
        """
        return self._f_validator
    
    def __call__(self, *args):
        """ Relation validation call

            Returns
            -----
            int (0 or 1)
                Returns weather the relation applies or not
        """
        return self._f_validator(*args)


class O2ORelation(object):
    
    # Object to object relations
    O2O_RELATIONS = OrderedDict([
        ("on", "is_on"),
        ("in", "is_in"),
        ("front", "is_front"),
        ("behind", "is_behind"),
        ("right_side", "is_right"),
        ("left_side", "is_left"),
        ("above", "is_above"),
        ("below", "is_below"),
        ("holding", "is_holding"),
        ("near", "is_near")
    ])
    # o2o_index = {_o: _j for _j, _o in enumerate(O2O_RELATIONS)}

    def __init__(self, *args):
        """ Constructor
        """
        self._rep = [] # Representation vector
        self.objs = args  # Objects to compute the relationship between
        # initializes all possible relations
        for _n, _func in self.O2O_RELATIONS.items():
            self.__dict__[_n] = Relation(
                _n, 
                getattr(self, _func)
            )
        # Compute rules
        self.evaluate()
        
    @property
    def rep(self):
        """ Relations getter
        """
        return self._rep
    
    def decode(self):
        """ Decode relationships for human interpretation
        """
        o_reps = [_o.__repr__() for _o  in self.objs]
        return [("_" 
                + list(self.O2O_RELATIONS.keys())[_i].upper()
                + "_").join(o_reps) \
            for _i, _r in enumerate(self._rep) if _r
        ]
    
    def __repr__(self):
        return '<' + ','.join(self.decode()) \
                +  ":" + super().__repr__().split('.')[-1]
    
    def evaluate(self):
        """ Evaluate over all registered relations,
            set which of them are valid with 1 and 0 otherwise,
            to return a one-hot vector.
        """
        # Call all rules
        _evals = [self.__dict__[_r]() \
            for _r in self.O2O_RELATIONS.keys()]
        # Set the relations representation
        self._rep = np.array(_evals)

    @staticmethod
    def zeros():
        """ Return zero vector with 
            O2O Relations cardinality

            Returns
            -----
            np.ndarray 
                Zero Vector
        """
        return np.zeros((len(O2ORelation.O2O_RELATIONS),))

    def default(self):
        """ Default Rule function
        """
        # print("Executing DEFAULT..")
        return 0.

    def is_on(self):
        """ ON rule validator: returns 1 in case
            the 1st object has greater Z and 
            the X and Y absolute difference is less or equal
            to the OBJ_W_SIZE, otherwise returns 0
        """
        # print("Executing IS_ON..")
        if self.objs[0].pos[2] > self.objs[1].pos[2]:
            _xdif = abs(self.objs[0].pos[0] - self.objs[1].pos[0])
            _ydif = abs(self.objs[0].pos[1] - self.objs[1].pos[1])
            if (_xdif <= OBJ_W_SIZE) and (_ydif <= OBJ_W_SIZE):
                return 1.
        return 0.
    
    def is_below(self):
        """ BELOW rule validator: 1 if the 1st object 
            has lower Z than the 2nd one, otherwise returns 0
        """
        # print("Executing IS_BELOW..")
        if self.objs[0].pos[2] < self.objs[1].pos[2]:
            return 1.
        return 0.
    
    def is_above(self):
        """ ABOVE rule validator: 1 if the 1st object 
            has greater Z than the 2nd one, otherwise returns 0
        """
        # print("Executing IS_ABOVE..")
        if self.objs[0].pos[2] > self.objs[1].pos[2]:
            return 1.
        return 0.
    
    def is_front(self):
        """ FRONT rule validator: 1 if the 1st object 
            has greater X than the 2nd one, otherwise returns 0
        """
        # print("Executing IS_FRONT..")
        if self.objs[0].pos[0] > self.objs[1].pos[0]:
            return 1.
        return 0.
    
    def is_behind(self):
        """ BEHING rule validator: 1 if the 1st object 
            has lower X than the 2nd one, otherwise returns 0
        """
        # print("Executing IS_BEHIND..")
        if self.objs[0].pos[0] < self.objs[1].pos[0]:
            return 1.
        return 0.
    
    def is_right(self):
        """ RIGHT rule validator: 1 if the 1st object 
            has greater Y than the 2nd one, otherwise returns 0
        """
        # print("Executing IS_RIGHT..")
        if self.objs[0].pos[1] > self.objs[1].pos[1]:
            return 1.
        return 0.
    
    def is_left(self):
        """ LEFT rule validator: 1 if the 1st object 
            has lower Y than the 2nd one, otherwise returns 0
        """
        # print("Executing IS_LEFT..")
        if self.objs[0].pos[1] < self.objs[1].pos[1]:
            return 1.
        return 0.

    def is_near(self):
        """ NEAR rule validator: 1 if the euclidean distance
            between the 2 objects is less or equal to the
            NEAR_DIST_TH and greater than HOLD_DIST_TH, 
            otherwise 0
        """
        # print("Executing IS_NEAR..")
        _dist = euclidean(self.objs[0].pos, self.objs[1].pos)
        if  _dist <= NEAR_DIST_TH and _dist > HOLD_DIST_TH:
            return 1.
        return 0.
    
    def is_holding(self):
        """ HOLD rule validator: 1 if the euclidean distance
            between the 2 objects is less or equal to the
            HOLD_DIST_TH, otherwise 0
        """
        # print("Executing IS_HOLDING..")
        if not self.objs[0].is_robot:
            return 0.
        _dist = euclidean(self.objs[0].pos, self.objs[1].pos)
        if  _dist <= HOLD_DIST_TH:
            return 1.
        return 0.

    def is_in(self, ):
        """ IN rule validator: 1 if the euclidean distance
            between the 1st object and the 2nd one is less 
            or equal to the IN_DIST_TH
        """
        # print("Executing IS_IN..")
        _dist = euclidean(self.objs[0].pos, self.objs[1].pos)
        if  _dist <= IN_DIST_TH:
            return 1.
        return 0.

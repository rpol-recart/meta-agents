"""
Test fixtures for Dependency Analysis Agent.
Contains code with various dependency patterns and issues.
"""

DEPENDENCY_SIMPLE = """
from module_a import function_a
from module_b import ClassB

def use_functions():
    result_a = function_a()
    obj_b = ClassB(result_a)
    return obj_b.process()
"""

DEPENDENCY_CIRCULAR_A = """
from dependency_circular_b import ClassB

class ClassA:
    def method_a(self):
        return ClassB().method_b()
"""

DEPENDENCY_CIRCULAR_B = """
from dependency_circular_a import ClassA

class ClassB:
    def method_b(self):
        return ClassA().method_a()
"""

DEPENDENCY_DEEP_HIERARCHY = """
class Base:
    def base_method(self):
        pass

class Level1(Base):
    def level1_method(self):
        pass

class Level2(Level1):
    def level2_method(self):
        pass

class Level3(Level2):
    def level3_method(self):
        pass

class Level4(Level3):
    def level4_method(self):
        pass

class Level5(Level4):
    def deep_method(self):
        return self.level4_method()
"""

DEPENDENCY_DIAMOND = """
class BaseClass:
    def method(self):
        print("BaseClass")

class LeftClass(BaseClass):
    def method(self):
        print("LeftClass")

class RightClass(BaseClass):
    def method(self):
        print("RightClass")

class DiamondClass(LeftClass, RightClass):
    pass
"""

DEPENDENCY_TRANSITIVE = """
from module_a import ClassA

class ClassB:
    def __init__(self):
        self.a = ClassA()
    
    def use_a(self):
        return self.a.operation()

class ClassC:
    def __init__(self):
        self.b = ClassB()
    
    def process(self):
        return self.b.use_a()
"""

DEPENDENCY_CONDITIONAL = """
class ServiceLocator:
    _services = {}
    
    @classmethod
    def register(cls, name, service):
        cls._services[name] = service
    
    @classmethod
    def get(cls, name):
        return cls._services.get(name)

def get_database():
    return ServiceLocator.get("database")

def get_cache():
    return ServiceLocator.get("cache")

def business_logic():
    db = get_database()
    cache = get_cache()
    return db.query(cache.get("key"))
"""

DEPENDENCY_DATABASE = """
class UserRepository:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def get_user(self, user_id):
        return self.db.query("SELECT * FROM users WHERE id = ?", user_id)
    
    def save_user(self, user):
        self.db.execute("INSERT INTO users VALUES (?)", user)

class OrderService:
    def __init__(self, user_repo, notification_service):
        self.users = user_repo
        self.notifications = notification_service
    
    def create_order(self, user_id, items):
        user = self.users.get_user(user_id)
        order = self._build_order(user, items)
        self.users.save_user(user)
        self.notifications.send(user.email, "Order created")
        return order
"""

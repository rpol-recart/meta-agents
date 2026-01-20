"""
Test fixtures for Pattern Recognition Agent.
Contains code with various design patterns and anti-patterns.
"""

PATTERN_FACTORY = """
class Vehicle:
    def drive(self):
        pass

class Car(Vehicle):
    def drive(self):
        return "Driving a car"

class Truck(Vehicle):
    def drive(self):
        return "Driving a truck"

class Bicycle(Vehicle):
    def drive(self):
        return "Riding a bicycle"

class VehicleFactory:
    @staticmethod
    def create_vehicle(vehicle_type):
        if vehicle_type == "car":
            return Car()
        elif vehicle_type == "truck":
            return Truck()
        elif vehicle_type == "bicycle":
            return Bicycle()
        raise ValueError(f"Unknown vehicle type: {vehicle_type}")
"""

PATTERN_OBSERVER = """
from abc import ABC, abstractmethod

class Observer(ABC):
    @abstractmethod
    def update(self, subject):
        pass

class Subject:
    def __init__(self):
        self._observers = []
    
    def attach(self, observer):
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer):
        self._observers.remove(observer)
    
    def notify(self):
        for observer in self._observers:
            observer.update(self)

class ConcreteObserver(Observer):
    def update(self, subject):
        print(f"Observer notified: {subject.state}")
"""

PATTERN_STRATEGY = """
class PaymentStrategy(ABC):
    @abstractmethod
    def pay(self, amount):
        pass

class CreditCardPayment(PaymentStrategy):
    def pay(self, amount):
        print(f"Paid {amount} via credit card")

class PayPalPayment(PaymentStrategy):
    def pay(self, amount):
        print(f"Paid {amount} via PayPal")

class PaymentContext:
    def __init__(self, strategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy):
        self._strategy = strategy
    
    def execute_payment(self, amount):
        return self._strategy.pay(amount)
"""

PATTERN_DECORATOR = """
class Coffee:
    def get_description(self):
        return "Coffee"
    
    def cost(self):
        return 5

class CoffeeDecorator(Coffee):
    def __init__(self, coffee):
        self._coffee = coffee
    
    def get_description(self):
        return self._coffee.get_description()
    
    def cost(self):
        return self._coffee.cost()

class MilkDecorator(CoffeeDecorator):
    def get_description(self):
        return self._coffee.get_description() + ", Milk"
    
    def cost(self):
        return self._coffee.cost() + 1

class SugarDecorator(CoffeeDecorator):
    def get_description(self):
        return self._coffee.get_description() + ", Sugar"
    
    def cost(self):
        return self._coffee.cost() + 0.5
"""

ANTIPATTERN_GOD_CLASS = """
class SystemManager:
    def __init__(self):
        self.users = []
        self.orders = []
        self.products = []
        self.invoices = []
        self.reports = []
    
    def add_user(self, user):
        self.users.append(user)
    
    def delete_user(self, user):
        self.users.remove(user)
    
    def create_order(self, order):
        self.orders.append(order)
    
    def cancel_order(self, order):
        self.orders.remove(order)
    
    def generate_report(self):
        pass
    
    def validate_invoice(self, invoice):
        pass
    
    def process_payment(self, payment):
        pass
    
    def send_email(self, recipient, message):
        pass
"""

ANTIPATTERN_LAVAS = """
def process_data(data):
    result = []
    for item in data:
        if item.active:
            processed = transform(item.value)
            if processed:
                result.append(processed)
    return result

def transform(value):
    if value > 0:
        return value * 2
    return None
"""

ANTIPATTERN_SPAGHETTI_CODE = """
def complex_function(data):
    if data:
        for i in range(len(data)):
            if data[i] > 0:
                for j in range(i):
                    if data[j] < data[i]:
                        temp = data[i]
                        data[i] = data[j]
                        data[j] = temp
    return data
"""

ANTIPATTERN_DUPLICATED_CODE = """
def calculate_area_rectangle(width, height):
    return width * height

def calculate_area_square(side):
    return side * side

def calculate_area_triangle(base, height):
    return base * height / 2

# Same formula repeated
def calculate_volume_rect_prism(width, height, depth):
    return width * height * depth
"""

"""
Test fixtures for Semantic Analyzer Agent.
Contains code with various semantic patterns and issues.
"""

SEMANTIC_SIMPLE_FUNCTION = '''
def calculate_total(items):
    """Calculate total price of items."""
    total = 0
    for item in items:
        total += item.price * item.quantity
    return total
'''

SEMANTIC_OBSCURE_LOGIC = """
def process_data(input_value):
    result = []
    for item in input_value:
        if item.status == 1:
            temp = transform(item.data)
            if temp > 100:
                result.append(temp)
    return sorted(result)
"""

SEMANTIC_MULTI_LAYER = """
class OrderProcessor:
    def __init__(self, validator, repository):
        self.validator = validator
        self.repository = repository
    
    def process(self, order_data):
        if not self.validator.validate(order_data):
            raise ValueError("Invalid order")
        
        order = self.repository.create(order_data)
        return self._execute_order(order)
    
    def _execute_order(self, order):
        # Complex business logic
        for item in order.items:
            item.reserved = True
        order.status = 'processing'
        return self.repository.save(order)
"""

SEMANTIC_DATA_FLOW = """
def user_registration_flow(user_input):
    # Input validation
    validated = validate_user_input(user_input)
    
    # Data transformation
    user_model = convert_to_user_model(validated)
    
    # Business logic
    created = create_user_account(user_model)
    
    # Notification
    send_welcome_email(created.email)
    
    return created
"""

SEMANTIC_INCONSISTENT_NAMING = """
class DataHelper:
    def do_stuff(self, x):
        return x * 2
    
    def process_it(self, data):
        for item in data:
            self.do_stuff(item)
    
    def xyz(self):
        return "confusing"
"""

import math
from jimmylabs.autograd.value import Value

def test_numerical_gradient():
    """
    Test comparing finite-difference gradients vs analytic gradients
    computed by Value.backward(). Proves that hand-understanding == autograd.
    """
    h = 1e-4
    
    # Simple expression: f(x, y) = (x * y) + relu(x)
    x_val = -2.0
    y_val = 3.0
    
    # Analytic calculation
    x = Value(x_val)
    y = Value(y_val)
    f = (x * y) + x.relu()
    f.backward()
    
    analytic_dx = x.grad
    analytic_dy = y.grad
    
    # Numeric calculation
    f_xy = (x_val * y_val) + max(0, x_val)
    f_xh_y = ((x_val + h) * y_val) + max(0, x_val + h)
    f_x_yh = (x_val * (y_val + h)) + max(0, x_val)
    
    numeric_dx = (f_xh_y - f_xy) / h
    numeric_dy = (f_x_yh - f_xy) / h
    
    # We allow a small tolerance due to finite difference approximation
    assert abs(analytic_dx - numeric_dx) < 1e-3, "Analytic dx does not match numeric dx"
    assert abs(analytic_dy - numeric_dy) < 1e-3, "Analytic dy does not match numeric dy"

def test_known_tiny_expression():
    """
    A known tiny expression backprops to hand-verified gradients.
    """
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    
    e = a * b
    d = e + c
    f = Value(-2.0)
    L = d * f
    L.backward()
    
    # Forward pass:
    # L = d * f => L = (a*b + c) * f
    # L = (2 * -3 + 10) * -2 = (4) * -2 = -8
    assert L.data == -8.0
    
    # Backward pass:
    # dL/df = d = 4.0
    assert f.grad == 4.0
    # dL/dd = f = -2.0
    # dL/dc = dL/dd * dd/dc = -2.0 * 1.0 = -2.0
    assert c.grad == -2.0
    # dL/de = dL/dd * dd/de = -2.0 * 1.0 = -2.0
    # dL/da = dL/de * de/da = -2.0 * b = -2.0 * -3.0 = 6.0
    assert a.grad == 6.0
    # dL/db = dL/de * de/db = -2.0 * a = -2.0 * 2.0 = -4.0
    assert b.grad == -4.0

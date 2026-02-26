from django import template

register = template.Library()

@register.filter
def multiply(value, arg):
    """Multiply the value by arg"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        try:
            return value * arg
        except:
            return 0
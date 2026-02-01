# Copyright (c) 2024, pUniFind Team
# This module initializes the model and task registries
# by importing the actual model and task implementations

def init_registries():
    """
    Initialize the model and task registries by importing the implementations.
    This ensures that all @register_model and @register_task decorators are executed.
    """
    # Import models to trigger registration
    try:
        from PepMS.models import ms_pep
    except ImportError:
        pass

    # Import tasks to trigger registration
    try:
        from PepMS.tasks import ms_pep_score, ms_pep_denovo
    except ImportError:
        pass


# Auto-initialize when this module is imported
init_registries()

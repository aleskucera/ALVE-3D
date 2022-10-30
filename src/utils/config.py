import logging

log = logging.getLogger(__name__)


def check_value(value: str, valid_values: list) -> None:
    """ Check if the value is valid
    :param value: The value to check
    :param valid_values: The list of valid values
    :return: True if the value is valid, False otherwise
    """
    if value not in valid_values:
        log.error(f'Invalid config value: {value}')
        exit(1)

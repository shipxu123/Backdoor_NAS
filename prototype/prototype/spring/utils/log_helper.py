import logging
import coloredlogs

logs = dict()

_level_style = {
    'critical': {
        'bold': True,
        'color': 'red'
    },
    'debug': {
        'color': 'green'
    },
    'dbg': {
        'color': 'green'
    },
    'err': {
        'color': 'red'
    },
    'info': {},
    'inf': {},
    'notice': {
        'color': 'magenta'
    },
    'spam': {
        'color': 'green',
        'faint': True
    },
    'success': {
        'bold': True,
        'color': 'green'
    },
    'verbose': {
        'color': 'blue'
    },
    'warning': {
        'color': 'yellow'
    },
    'wrn': {
        'color': 'yellow'
    }
}


def init_log(name, level=logging.INFO, rank=True):
    if (name, level) in logs:
        return logs[(name, level)]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if rank:
        from prototype.spring.utils.dist_helper import get_rank
        logger.addFilter(lambda record: get_rank() == 0)

    coloredlogs.install(
        fmt='%(asctime)s %(levelname)s %(filename)s#%(lineno)d %(message)s',
        level_styles=_level_style,
        level=level,
        logger=logger)

    logger.propagate = False
    logs[(name, level)] = logger

    return logger


class SpringLogger(object):
    def __init__(self, rank=True):
        self.logger = None
        self.rank = rank

    def __getattr__(self, obj):
        if self.logger is None:
            self.logger = init_log('global', logging.INFO, self.rank)
        return getattr(self.logger, obj)


default_logger = SpringLogger()

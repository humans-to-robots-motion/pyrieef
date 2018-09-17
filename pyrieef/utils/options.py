import optparse


def add_boolean_options(parser, options):
    for o in options:
        parser.add_option('--' + o, action='store_true', default=False)

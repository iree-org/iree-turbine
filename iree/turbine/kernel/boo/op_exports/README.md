Signatures of BOO Operations

To add a new op, define its signature and parser class by subclassing `OpSignature` and `OpCLIParser` defined in ../exports and list the op with its key in `registry.py`. The key be a unique string that does not appear in any string accepted by the parser of another op.

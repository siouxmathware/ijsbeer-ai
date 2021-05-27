"""
Definitions of global error types for the server.
"""


class BadRequest(Exception):
    """Exception type to be raised when the request is invalid"""
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        """
        :param message: Message to be presented to the API use
        :param status_code: Status code to be yielded by the server
        :param payload: Payload of the operation, currently unused
        """
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        """
        :return: A Dict containing the error message and None for the 'results' field
        """
        rv = dict(self.payload or ())
        rv['message'] = self.message
        rv['results'] = None
        return rv


class InternalServerError(Exception):
    """Exception type to be raised when there is an error in any part of the AI pipeline"""
    status_code = 500

    def __init__(self, message, status_code=None, payload=None):
        """
        :param message: Message to be presented to the API use
        :param status_code: Status code to be yielded by the server
        :param payload: Payload of the operation, currently unused
        """
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        """
        :return: A Dict containing the error message and None for the 'results' field
        """
        rv = dict(self.payload or ())
        rv['message'] = self.message
        rv['results'] = None
        return rv

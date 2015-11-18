import abc

class catalog(object):
  __metaclass__ = abc.ABCMeta

  def exists(self):
    return False


  @abc.abstractmethod
  def conn(self, host='localhost'):
    """
    Method to connect to server on given host
    """
    pass


  @abc.abstractmethod
  def start(self):
    """
    Method to start up service
    """
    pass

  @abc.abstractmethod
  def stop(self):
    """
    Method to stop service
    """
    pass



  @abc.abstractmethod
  def load(self, key):
    """
    Load a data element
    """
    pass


  @abc.abstractmethod
  def save(self, key):
    """
    Save a data element
    """
    pass


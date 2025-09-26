import json
import logging
import os
from os import path

# Configure logging
logger = logging.getLogger(__name__)


def save(filename, data, overwrite=False):
    """
    Save an object to a file.

    Parameters
    ----------
    filename : str
        The name of the file where the object will be saved. Can include the
        .gwref extension or not. Can include a path.
    data : dict
        The data to save, typically a dictionary containing the object and its metadata.
    overwrite : bool, optional
        If True, overwrite the file if it already exists. Default is False.

    Returns
    -------
    Message
        A message indicating the success or failure of the save operation.
    """
    # Check the filename extension
    ext = os.path.splitext(filename)[1]
    if ext == "":
        ext = ".gwref"
    elif ext not in [".gwref"]:
        logger.error(f"Unsupported file extension: {ext}. Expected '.gwref'.")
        raise ValueError(f"Unsupported file extension: {ext}. Expected '.gwref'.")
    filename = path.splitext(filename)[0] + ext

    # Check if the file already exists
    if os.path.exists(f"{filename}") and not overwrite:
        # If the file exists and overwrite is False, log a warning and return
        logger.warning(
            f"The file '{filename}' already exists. To overwrite the existing file"
            f" set the argument 'overwrite' to True."
        )
        return False

    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

    return True


def load(filename):
    """
    Load an object from a file.

    Parameters
    ----------
    filename : str
        The name of the file from which the object will be loaded.

    Returns
    -------
    object
        The loaded object.
    """
    ext = path.splitext(filename)[1]
    if ext == "":
        filename += ".gwref"
    elif ext not in [".gwref"]:
        logger.error(f"Unsupported file extension: {ext}. Expected '.gwref'.")
        raise ValueError(f"Unsupported file extension: {ext}. Expected '.gwref'.")

    if not os.path.exists(filename):
        # Check if it is in the package examples directory
        if path.exists(path.join(path.dirname(__file__), "..", "examples", filename)):
            filename = path.join(path.dirname(__file__), "..", "examples", filename)
        else:
            logger.error(f"The file {filename} does not exist.")
            raise FileNotFoundError(f"The file {filename} does not exist.")

    with open(filename) as file:
        data = json.load(file)

    logger.debug(f"Object loaded from {filename}")
    return data

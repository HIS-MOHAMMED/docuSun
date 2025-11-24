

def clean_extra_whitespaces(text):
    """
    Clean extra whitespaces from provided text

    Parameters
    - text: A string representing the text to be cleaned

    Returns:
    - A string with extra whitespaces removed
    """

    return ' '.join(text.split())


def group_broken_paragraphs(text):
    """
    Groups broken paragraphs in the  provided text

    Parameters:
    - text: A string representing the text to be processed

    Returns:
    - A string with broken paragraphs groubed
    """

    return text.replace("\n", " ").replace("\r", " ")
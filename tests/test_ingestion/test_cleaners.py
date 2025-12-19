from unittest.mock import patch
from engine.ingestion.cleaners import clean_extra_whitespaces, group_broken_paragraphs



def test_clean_extra_whitespaces():
    # Arrange

    # create a text with whitespaces
    text = 'Hello    Hisham,     welcome     to      docuSun.'

    # Act
    result = clean_extra_whitespaces(text)

    # Assert
    expected = 'Hello Hisham, welcome to docuSun.'
    assert result == expected

   
def test_group_broken_paragraphs():
    # Arrange

    # create a text with broken paragraphs
    text  = 'With docuSun,\nno worry about\rprivacy.'

    # Act
    result = group_broken_paragraphs(text)

    # Assert
    expected = 'With docuSun, no worry about privacy.'

    assert result == expected
        

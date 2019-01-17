TRUE_CAT = '1'
FALSE_CAT = '0'


def predict_category(predictions, category='crying_baby', strategy='Panic'):
    """ Is the category in predictions

    Strategy plan
    Confident - all category the same as in the max probability place
    50/50 - as min as half in the max probability place
    Panic(default) - even if selected category present in the second probability place

    :param
    predictions: list of Dict with {category: probabilities}
    category: for seek in prediction
    strategy:
    :return: '1' - true, '0' - false
    """
    return {
        'Full': TRUE_CAT if len([1 for prediction in predictions for cat in [list(prediction.keys())[0]] if cat == category]) == len(predictions) else FALSE_CAT,
        'Half': TRUE_CAT if len([1 for prediction in predictions for cat in [list(prediction.keys())[0]] if cat == category]) > len(predictions) / 2.0 else FALSE_CAT,
        'Panic': TRUE_CAT if len([1 for prediction in predictions for cat in list(prediction.keys()) if cat == category]) > 0 else FALSE_CAT
    }[strategy]

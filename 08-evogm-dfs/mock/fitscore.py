def score(it, output, fr):
    return dict(correct=output == "OK", truncated=fr == "length")

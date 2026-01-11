import numpy as np


def normalize(x, x_min, x_max):
    """
    Min-max normalization to scale values into [0, 1]
    """
    return (x - x_min) / (x_max - x_min)


def fitness_function(X):
    """
    Fitness function for a student chromosome

    X = [x1, x2, x3, x4, x5, x6, x7]
    """
    # --------- Realistic bounds for each gene ---------
    bounds = {
        "cgpa": (0.0, 4.0),
        "internship": (0, 12),  # months
        "attendance": (0, 100),  # %
        "professional_development": (0, 20),  # certificates
        "peer_evaluation": (0, 10),  # score
        "stress_tolerance": (0, 16),  # productive hours/day
        "deadline_penalty": (0, 100),  # weighted penalty
    }

    # --------- Normalization ---------
    x1 = normalize(X[0], *bounds["cgpa"])
    x2 = normalize(X[1], *bounds["internship"])
    x3 = normalize(X[2], *bounds["attendance"])
    x4 = normalize(X[3], *bounds["professional_development"])
    x5 = normalize(X[4], *bounds["peer_evaluation"])
    x6 = normalize(X[5], *bounds["stress_tolerance"])
    x7 = normalize(X[6], *bounds["deadline_penalty"])

    # --------- Weights ---------
    w1, w2, w3 = 0.25, 0.15, 0.15
    w4, w5, w6 = 0.10, 0.10, 0.10
    w7 = 0.20  # penalty weight

    # --------- Fitness calculation ---------
    fitness = (
        w1 * x1
        + w2 * x2
        + w3 * x3
        + w4 * x4
        + w5 * x5
        + w6 * x6
        - w7 * (x7**2)  # non-linear penalty
    )

    return fitness


# Example usage
if __name__ == "__main__":
    student = [
        3.6,  # CGPA
        6,  # internship months
        90,  # attendance %
        5,  # professional courses
        8.5,  # peer evaluation
        10,  # stress tolerance (hours/day)
        15,  # deadline penalty
    ]

    score = fitness_function(student)
    print("Fitness Score:", round(score, 4))

#!/usr/bin/env python
import csv
import numpy as np
import matplotlib.pyplot as plt


def K(u):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-(np.power(u, 2) / 2))


def dist(p, q):
    return np.sqrt(np.power(p[0] - q[0], 2) + np.power(p[1] - q[1], 2))


def kernel_density_estimation(t):
    return (1 / 5) * K(t / 5)


def calculate_value(x, y, points):
    distances = [dist([x, y], p) for p in points]
    point_values = [kernel_density_estimation(v) for v in distances]
    return sum(point_values)


def main():
    points = []
    with open('point_data.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            points.append([float(i) for i in row])
    points = np.array(points)
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(40, 270)
    ax.set_ylim(40, 160)
    plt.savefig('scatter.png')
    # plt.show()
    plt.close()

    x_min = y_min = 40
    x_max = 270
    y_max = 160

    x = np.arange(x_min, x_max)
    y = np.arange(y_min, y_max)
    Z = np.zeros((x.shape[0], y.shape[0]))

    for x_coords in x:
        for y_coords in y:
            Z[x_coords - x_min, y_coords - y_min] = calculate_value(x_coords, y_coords, points)

    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.imshow(Z.T, origin='lower', cmap='Blues', extent=[x_min, x_max, y_min, y_max])
    plt.savefig('density.png')
    plt.show()


if __name__ == '__main__':
    main()

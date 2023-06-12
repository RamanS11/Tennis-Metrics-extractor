import numpy as np


class Line:

    def __int__(self, point, vector):

        self.pointOnLine = None
        self.intersectionPoint = None
        self.u = point
        self.v = vector

    def fromRhoTheta(self, rho, theta):
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        p1 = ((x0 + 2000 * (-b)), y0 + 2000 * (-b))
        p2 = ((x0 - 2000 * (-b)), y0 - 2000 * (-b))
        return self.fromTwoPoints(p1, p2)

    def fromTwoPoints(self, p1: np.ndarray, p2: np.ndarray):
        p1 = np.array((p1[0], p1[1]))
        p2 = np.array((p2[0], p2[1]))
        vec = np.array((p2[0] - p1[0], p2[1] - p1[1]))

        return Line().__int__(p1, vec)

    def getPoint(self):
        return self.u

    def getVector(self):
        return self.v

    def getDistance(self, point):
        self.pointOnLine = self.getPointOnLineClosestTo(point)
        dx = point[0] - self.pointOnLine[0]
        dy = point[1] - self.pointOnLine[1]
        dist = np.sqrt(dx * dx + dy * dy)
        return dist

    def getPointOnLineClosestTo(self, point):
        n, c = self.toImplicit()
        q = c - n.dot(point)
        return point - q * n

    def isDuplicated(self, otherLine):
        n1, c1 = self.toImplicit()
        n2, c2 = otherLine.toImplicit()
        dot = np.fabs(n1.dot(n2))
        dotThreshold = np.cos(1 * np.pi / 180)
        if dot > dotThreshold and np.fabs(np.fabs(c1) - np.fabs(c2) < 10):
            return True
        else:
            return False

    def toImplicit(self):
        x, y = self.u
        n = np.array([-y, x])
        c = n.dot(self.u)
        return n, c

    def isVertical(self):
        n, c = self.toImplicit()
        check1 = np.fabs(np.arctan2(n[1], n[0]) < 65 * np.pi / 180)
        check2 = np.fabs(np.arctan2(-n[1], -n[0]) < 65 * np.pi / 180)

        return check1 and check2

    def computeIntersectionPoint(self, line):
        x = line.getPoint() - self.u
        d1 = np.copy(self.v)
        d2 = line.getVector()
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if np.abs(cross) < 10e-8:
            return False, self.intersectionPoint
        t1 = (x[0] * d1[1] - x[1] * d2[0]) / cross
        self.intersectionPoint = self.u + d1 * t1
        return True, self.intersectionPoint

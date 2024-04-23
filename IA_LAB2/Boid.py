from IA_LAB2.Vector import Vector


class Boid:
    def __init__(self, pos, vel, acc, max_speed, max_force, separation_dist, alignment_dist, cohesion_dist, boids):
        self.pos = Vector(pos)
        self.vel = Vector(vel)
        self.acc = Vector(acc)
        self.max_speed = max_speed
        self.max_force = max_force
        self.separation_dist = separation_dist
        self.alignment_dist = alignment_dist
        self.cohesion_dist = cohesion_dist
        self.boids = boids

    def apply_force(self, force):
        self.acc += force

    def update(self):
        self.pos += self.vel
        self.vel += self.acc
        self.vel.limit(self.max_speed)
        self.acc *= 0  # Reset acceleration after each update

    def separation(self, boids):
        steer = Vector([0, 0])
        total = 0

        for other in boids:
            if other != self:
                distance = self.pos.distance(other.pos)
                if 0 < distance < self.separation_dist:
                    diff = self.pos - other.pos
                    diff.normalize()
                    diff /= distance  # Weight by distance
                    steer += diff
                    total += 1

        if total > 0:
            steer /= total
            steer.normalize()
            steer *= self.max_speed
            steer -= self.vel
            steer.limit(self.max_force)

        return steer

    def align(self, boids):
        steer = Vector([0, 0])
        total = 0

        for other in boids:
            if other != self:
                distance = self.pos.distance(other.pos)
                if 0 < distance < self.alignment_dist:
                    steer += other.vel
                    total += 1

        if total > 0:
            steer /= total
            steer.normalize()
            steer *= self.max_speed
            steer -= self.vel
            steer.limit(self.max_force)

        return steer

    def cohesion(self, boids):
        steer = Vector([0, 0])
        total = 0
        center_of_mass = Vector([0, 0])

        for other in boids:
            if other != self:
                distance = self.pos.distance(other.pos)
                if 0 < distance < self.cohesion_dist:
                    center_of_mass += other.pos
                    total += 1

        if total > 0:
            center_of_mass /= total
            steer = center_of_mass - self.pos
            steer.normalize()
            steer *= self.max_speed
            steer -= self.vel
            steer.limit(self.max_force)

        return steer

    def flock(self, boids):
        separation_force = self.separation(boids)
        alignment_force = self.align(boids)
        cohesion_force = self.cohesion(boids)

        # Adjust the strength of each behavior based on the paper's recommendations
        separation_force *= 1.5  # Increase separation strength
        alignment_force *= 1.0  # Keep alignment strength unchanged
        cohesion_force *= 1.0  # Keep cohesion strength unchanged

        self.apply_force(separation_force)
        self.apply_force(alignment_force)
        self.apply_force(cohesion_force)

        self.update()

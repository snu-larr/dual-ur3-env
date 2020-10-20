class Wall3D:
    def __init__(self, x_center, y_center, z_center,  x_thickness, y_thickness, z_thickness, min_dist):
        self.min_x = x_center - x_thickness - min_dist
        self.max_x = x_center + x_thickness + min_dist
        self.min_y = y_center - y_thickness - min_dist
        self.max_y = y_center + y_thickness + min_dist
        self.min_z = z_center - z_thickness - min_dist
        self.max_z = z_center + z_thickness + min_dist

        # self.endpoint1 = (x_center+x_thickness, y_center+y_thickness)
        # self.endpoint2 = (x_center+x_thickness, y_center-y_thickness)
        # self.endpoint3 = (x_center-x_thickness, y_center-y_thickness)
        # self.endpoint4 = (x_center-x_thickness, y_center+y_thickness)

    def contains_point(self, point):
        return (self.min_x < point[0] < self.max_x) and (self.min_y < point[1] < self.max_y) and (self.min_z < point[2] < self.max_z)

    def contains_points(self, points):
        return (self.min_x < points[:,0]) * (points[:,0] < self.max_x) \
               * (self.min_y < points[:,1]) * (points[:,1] < self.max_y) \
                   * (self.min_z < points[:,2]) * (points[:,2] < self.max_z)

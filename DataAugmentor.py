import Augmentor

p = Augmentor.Pipeline("dataset-resized/cardboard")

p.rotate90(probability=0.5)
p.rotate270(probability=0.5)
p.flip_left_right(probability=0.8)
p.flip_top_bottom(probability=0.3)
p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
# p.flip_left_right(probability=0.5)
# p.flip_top_bottom(probability=0.5)
# p.crop_random(probability=1, percentage_area=0.5)
# p.resize(probability=1.0, width=120, height=120)
# p.skew_tilt(probability = 0.5)
# p.skew_left_right(probability = 0.5)
# p.skew_top_bottom(probability = 0.5)
# skew_corner(probability = 0.5)


p.sample(10000)

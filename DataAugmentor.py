import Augmentor

# p = Augmentor.Pipeline("dataset-resized/cardboard")
# p = Augmentor.Pipeline("dataset-resized/glass")
# p = Augmentor.Pipeline("dataset-resized/metal")
# p = Augmentor.Pipeline("dataset-resized/paper")
# p = Augmentor.Pipeline("dataset-resized/plastic")


# p.rotate90(probability=0.5)
# p.rotate270(probability=0.5)
p.rotate_random_90(probability = 0.5)
p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.5)
p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=6)
# p.flip_left_right(probability=0.5)
# p.flip_top_bottom(probability=0.5)
# p.crop_random(probability=1, percentage_area=0.5)

p.skew_tilt(probability = 0.5)
p.skew_left_right(probability = 0.5)
p.skew_top_bottom(probability = 0.5)
p.skew_corner(probability = 0.5)
# p.crop_random(probability=1, grid_width=4, grid_height=4)

p.resize(probability=1.0, width=512, height=384)
p.sample(10000)

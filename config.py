
PASTEL = 'pastel'
GAUSSIAN_NOISE = 'gnl'
ADVERSARIAL_NOISE = 'anl'
PASTEL_GAUSSIAN_NOISE = 'pastel_gnl'
PASTEL_ADVERSARIAL_NOISE = 'pastel_anl'


# Group setup
PASTEL_CONFIG = [PASTEL, PASTEL_GAUSSIAN_NOISE, PASTEL_ADVERSARIAL_NOISE]
ADVERSARIAL_CONFIG = [ADVERSARIAL_NOISE, PASTEL_ADVERSARIAL_NOISE]
GAUSSIAN_CONFIG = [GAUSSIAN_NOISE, PASTEL_ADVERSARIAL_NOISE]
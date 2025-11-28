// Currently supported species for ReID
export const ACTIVE_SPECIES = ['stoat'] as const;

// Species coming in future updates (alphabetical order)
export const FUTURE_SPECIES = [
    'bird',
    'cat',
    'deer',
    'dog',
    'ferret',
    'goat',
    'hedgehog',
    'kea',
    'kiwi',
    'lagomorph',
    'livestock',
    'parakeet',
    'pig',
    'possum',
    'pukeko',
    'rodent',
    'takahe',
    'tomtit',
    'tui',
    'wallaby',
    'weasel',
    'weka',
    'yellow eyed penguin'
] as const;

// All species combined
export const ALL_SPECIES = [...ACTIVE_SPECIES, ...FUTURE_SPECIES] as const;

export const DEFAULT_SPECIES = 'stoat';

export type ActiveSpecies = typeof ACTIVE_SPECIES[number];
export type FutureSpecies = typeof FUTURE_SPECIES[number];
export type AllSpecies = typeof ALL_SPECIES[number];

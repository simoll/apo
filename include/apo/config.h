#ifndef APO_CONFIG_H
#define APO_CONFIG_H

const bool Verbose = false;

#define IF_VERBOSE if (Verbose)

// expensive consistency checks
#ifdef NDEBUG
#define IF_DEBUG if (false)
#else
#define IF_DEBUG if (true)
#endif

#endif // APO_CONFIG_H

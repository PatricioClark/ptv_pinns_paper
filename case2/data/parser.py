# python3

# Parse TecPlot2 files

import sys

tidx  = 0
write = False

# Open file
with open('TecPlot2.dat') as data:
    [next(data) for _ in range(3)]

    # Loop through lines
    for line in data:
        if line.startswith('ZONE T') and write:
            dump.close()
            tidx += 1
            write = False

        # Write lines
        if write:
            dump.write(line)

        if   line.startswith('DATAPACKING'):
            dump  = open(f'velos.{tidx:04}.dat', 'w') 
            write = True

print('Nt =', tidx)

from matrices_generator import readgri, verify


def main():
    """Report max |E_e| for every mesh in the submission.

    E_e = sum_i n_ei^outward * l_ei should be the zero vector on each element,
    so the reported maximum is a measure of round-off only.
    """
    meshes = [
        ('test mesh', 'gri/test.gri'),
        ('coarse', 'gri/all.gri'),
        ('locally refined', 'gri/smoothed_local_all.gri'),
        ('uniform 8k', 'gri/refinement_uniform_all_8k.gri'),
        ('uniform 32k', 'gri/refinement_uniform_all_32k.gri'),
        ('uniform 128k', 'gri/refinement_uniform_all_128k.gri'),
    ]

    print(f"{'mesh':<18}{'elements':>10}{'max |E_e|':>14}")
    for label, fname in meshes:
        # report failures rather than skipping them, so a missing or broken
        # mesh can never be mistaken for a passing row
        try:
            Mesh = readgri(fname)
            print(f'{label:<18}{len(Mesh["E"]):>10}{verify(fname, Mesh):>14.3e}')
        except Exception as e:
            print(f'{label:<18}{"-":>10}{"FAILED":>14}   {type(e).__name__}: {e}')


if __name__ == "__main__":
    main()

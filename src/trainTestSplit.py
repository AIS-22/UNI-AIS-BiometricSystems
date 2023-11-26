import splitfolders


def main():
    datasetName = 'PLUS'
    # train test split
    splitfolders.ratio("data_prepared/" + datasetName, output="data/" + datasetName,
                       seed=42, ratio=(.8, .2), group_prefix=None, move=False)  # default values


if __name__ == '__main__':
    main()

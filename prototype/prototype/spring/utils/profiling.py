import csv
import os
import os.path as osp
import argparse

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input file name")
    parser.add_argument("-o", "--output", help="output dir")
    return parser.parse_args()


def prof_summary(out_csv):
    summary = {}
    kernel_total, t_total = 0, 0
    with open(out_csv) as fin:
        f_csv = csv.reader(fin)
        headings = next(f_csv)
        for r in f_csv:
            kernel = r[-3]
            t = int(r[-1])
            kernel_total += 1
            t_total += t
            if kernel not in summary:
                summary[kernel] = [0, 0]
            summary[kernel][0] += 1
            summary[kernel][1] += t
    template = "{:50.50} {:10.10} {:10.10}"
    summary_sorted = sorted(summary.items(), key=lambda kv:(kv[1][1]), reverse=True)
    print(template.format("kernel", "count", "time"))

    for k, v in summary_sorted:
        print(template.format(k, str(v[0]), str(v[1]/1000000)) + 'ms')
        # print("{} {} {}ms".format(k, v[0], v[1]/1000000))
    print("=============================================================================")
    print(template.format("Total", str(kernel_total), str(t_total / 1000000) + 'ms'))


def parse_db(input_file, output_dir='.'):
    if not osp.isdir(output_dir):
        os.mkdir(output_dir)
    out_dict_file = osp.join(output_dir, 'net.dict')
    cmd = 'python -m pyprof.parse {} > {}'.format(input_file, out_dict_file)
    print("begin to parse {}".format(input_file))
    os.system(cmd)
    print("finish parse {}".format(input_file))
    print("generate csv...")
    out_csv_file = osp.join(output_dir, 'net.csv')
    cmd = "python -m pyprof.prof --csv {} > {}".format(out_dict_file, out_csv_file)
    os.system(cmd)
    prof_summary(out_csv_file)

if __name__ == '__main__':
    args = parse_arg()
    parse_db(args.input, args.output)
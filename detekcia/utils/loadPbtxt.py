class LoadPbtxt:
    def parser(self, path: str):
        tmp = {}
        with open(path, 'r') as fh:
            graph_str = fh.readlines()

            idx, disp_name = None, None
            for line in graph_str:
                if line.__contains__("id"):
                    idx = int(line.split(":")[-1].strip())
                if line.__contains__("display_name"):
                    disp_name = line.split(":")[-1].strip().split("\"")[1]

                if disp_name and idx:
                    tmp[idx] = disp_name
                    idx, disp_name = None, None
            fh.close()
        return tmp

if __name__ == '__main__':
    LoadPbtxt().parser(path="data/labels/mscoco_label_map.pbtxt")
import numpy
import glob

path_input_root = "../../argodataset_trackerformat/*"
path_inputs = glob.glob(path_input_root)
path_output = "../../tracker_output"

logid_to_city_dict = {
    "089813dd-b5df-30ea-aaa7-fd5a9acf5302": "MIA",
    "38a7c63f-3304-3e76-9481-8aa1c745d18c": "MIA",
    "10b3a1d8-e56c-38be-aaf7-ef2f862a5c4e": "MIA",
    "c9d6ebeb-be15-3df8-b6f1-5575bea8e6b9": "MIA",
    "bae67a44-0f30-30c1-8999-06fc1c7ab80a": "MIA",
    "9aea22e0-70d3-34e6-a0b2-b7f6afdaaa27": "DTW",
    "5c251c22-11b2-3278-835c-0cf3cdee3f44": "MIA",
    "38b2c7ef-069b-3d9d-bbeb-8847b8c89fb6": "PIT",
    "da3d8357-54b1-321f-8efa-a0332668096f": "PIT",
    "e07be70a-db1f-3731-b912-b0439065f766": "MIA",
    "76a9f363-bdc5-330b-94d5-05c6e8f29bf6": "MIA",
    "e4adb13f-ec05-373d-ae9a-cd3c683eb869": "MIA",
    "84c35ea7-1a99-3a0c-a3ea-c5915d68acbc": "MIA",
    "e7403a92-aabf-3354-af97-b20bea479d7d": "MIA",
    "577ea60d-7cc0-34a4-a8ff-0401e5ab9c62": "MIA",
    "693c4b41-2df6-3961-851b-3c2ddf5ea227": "MIA",
    "efb48719-7c42-31da-b203-0dd4eed633dc": "MIA",
    "609bd5f8-28d2-3965-a583-14e0f9752aaa": "MIA",
    "649750f3-0163-34eb-a102-7aaf5384eaec": "MIA",
    "033669d3-3d6b-3d3d-bd93-7985d86653ea": "PIT",
    "c28501ef-cf11-3def-b358-ecd98d1284ae": "PIT",
    "df30a5c0-b251-3546-8da3-7ae4503b0ab1": "MIA",
    "aeb73d7a-8257-3225-972e-99307b3a5cb0": "MIA",
    "a2139885-9169-3ac8-a4ca-337e1c9bf8f4": "MIA",
    "f5ced2e6-de7a-3167-8ae6-174b7d311ac7": "MIA",
    "0d2ee2db-4061-36b2-a330-8301bdce3fe8": "PIT",
    "ff5c497e-767a-3b1a-961d-a40e69cd122e": "PIT",
    "6162d72f-2990-3a30-9bba-19bbd882985c": "MIA",
    "6db21fda-80cd-3f85-b4a7-0aadeb14724d": "MIA",
    "b955891c-5a6e-32fe-9bca-4edbcc3e5000": "PIT",
    "313b45e6-ef2e-37ce-aa26-f4b03fe685f4": "MIA",
    "9da4ca63-f524-3b38-8c8b-624f17518574": "MIA",
    "dd6ab742-d656-36ad-876c-fa1449710926": "MIA",
    "6593dead-ead2-31d2-b6d8-5d7975c82d2b": "PIT",
    "3ced8dba-62d0-3930-8f60-ebeea2feabb8": "MIA",
    "a2b55686-4b10-383d-9f83-2f5b29d89c67": "PIT",
    "e3dacee8-2840-3a97-92bc-497c1fea2a42": "PIT",
}

# sudo python3.6 run_tracking_multi_KF_nogt.py
# --path_input=../../argodataset_trackerformat/033669d3-3d6b-3d3d-bd93-7985d86653ea --path_output=../../tracker_output/
# --dbscan_eps=1.2 --min_point_num=50  --ground_removal_th=0.4  --city_name='PIT'  --ground_level=0.4 --use_map_roi=True

# ('_roi_%d_lane_%d_fixbbox_%d') %(use_map_roi,use_map_lane,fix_bbox_size )
use_map_roi = False
use_map_lane = False
fix_bbox_size = True
text_file = open(
    "run_tracker_roi_%d_lane_%d_fixbbox_%d_inverse.sh"
    % (use_map_roi, use_map_lane, fix_bbox_size),
    "w",
)


for i in range(len(path_inputs) - 1, 0, -1):
    print(path_inputs[i])
    if path_inputs[i].split("/")[-1] in logid_to_city_dict:

        city_name = logid_to_city_dict[path_inputs[i].split("/")[-1]]
    else:
        print("city name not found!")
        continue

    command = (
        "sudo python3.6 run_tracking_multi_KF_nogt.py --path_input=%s --path_output=%s --dbscan_eps=1.2 --min_point_num=50  --ground_removal_th=0.4  --city_name=%s  --ground_level=0.4 "
        % (path_inputs[i], path_output, city_name)
    )

    if use_map_roi:
        command += " --use_map_roi=True "
    if use_map_lane:
        command += " --use_map_lane=True "

    if fix_bbox_size:
        command += " --fix_bbox_size=True "

    print(command)

    text_file.write(command + "\n")

text_file.close()

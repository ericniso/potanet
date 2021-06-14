#!/bin/bash

cd model

ROOT_DIR=/data
IMZML_ROI_ROOT_DIR=$ROOT_DIR/imzML_roi_single_spectra_raw
IMZML_ROI_EXTRACTED_ROOT_DIR=$ROOT_DIR/imzML_roi_single_spectra_extracted

mkdir -p $IMZML_ROI_EXTRACTED_ROOT_DIR

echo "----- Training"

mkdir -p $IMZML_ROI_EXTRACTED_ROOT_DIR/training

# ROI
PATIENTS=("213" "250" "262" "268" "302" "308" "381" "384" "442" "475" "565" "598" "647" "922" "992" "995" "1012" "1046" "1047" "1076" "1083" "1144"\
    "1145" "1208" "1126_exvivo" "1187_exvivo" "209" "288" "289" "563" "566" "597" "602" "612" "759" "851" "870" "891" "893" "928" "1013" "1064" "1085" "1127" "1141" "1142"\
    "1157" "1204" "1206" "1259" "1321" "1322" "241" "853" "1058" "1062" "1196" "1207")

# WHOLE
PATIENTS_WHOLE=("209" "262" "268" "278" "288" "289" "308" "316" "475" "525" "563" "566" "597" "598" "602" "612" "759" "851" "870" "891" "893" "928" "1013" "1046" "1047" "1064" "1083"\
    "1085" "1127" "1141" "1142" "1144" "1145" "1146" "1150" "1157" "1204" "1206" "1208" "1234" "1237" "1259" "1321" "1322" "213" "241" "250" "442" "853" "992" "995" "1012"\
    "1062" "1076" "1147" "I18_18417_exvivo" "I18_4981_exvivo" "6075_exvivo" "5883_3_exvivo" "992_exvivo" "995_exvivo" "I19_1033_exvivo" "1126_exvivo" "I19_11112_exvivo")

for P in ${PATIENTS[@]}
do
    # echo $P
    # python3 parser.py $IMZML_ROI_ROOT_DIR/training/"$P.imzML" $IMZML_ROI_EXTRACTED_ROOT_DIR/training/$P

    if [ -f $IMZML_ROI_ROOT_DIR/training/"$P.imzML" ] && [ -f $IMZML_ROI_ROOT_DIR/training/"$P.ibd" ]; then
        # echo $P
        python3 parser.py $IMZML_ROI_ROOT_DIR/training/"$P.imzML" $IMZML_ROI_EXTRACTED_ROOT_DIR/training/$P
    fi

    # if [ ! -f $IMZML_ROI_ROOT_DIR/training/"$P.imzML" ]; then
    #     echo "$P.imzML"
    # fi

    # if [ ! -f $IMZML_ROI_ROOT_DIR/training/"$P.ibd" ]; then
    #     echo "$P.ibd"
    # fi
done

echo "----- Validation"

mkdir -p $IMZML_ROI_EXTRACTED_ROOT_DIR/validation

# ROI
PATIENTS=("278" "290" "316" "387" "436" "440" "520" "525" "609" "621" "935" "987" "1075" "1081" "1082" "1084" "1122" "1123" "1126" "1147" "1149" "1156" "1172"\
    "260" "446" "562" "564" "864" "894" "923" "927" "1007" "1029" "1048" "1096" "1184" "1201" "1209" "1239" "297" "380" "509" "521" "683"\
    "762" "869" "890" "892" "1051" "1069" "1180" "1200" "1242" "1248" "1257" "1352" "1353" "1187" "1188" "1202" "1234" "1283" "1294" "1328" "1331"\
    "267" "300" "383" "513" "516" "613" "624" "686" "703" "785" "793" "839" "925" "945" "983" "985" "1006" "1010" "1023" "1050" "1059" "1072" "1074"\
    "1077" "1078" "1082" "1097" "1105" "1139" "1140" "1148" "1185" "1186" "1197" "1198" "1203" "1205" "1232" "1235" "1236" "1238" "1243" "1256"\
    "1258" "1260" "1308" "1311" "1324" "1326" "1330" "1333" "1339" "1346" "1347" "1348" "1349" "264" "265" "295" "552" "567" "783" "1005" "1014"\
    "1068" "1136" "1247" "1338" "1351")

# WHOLE
PATIENTS_WHOLE=("260" "302" "380" "381" "384" "387" "436" "440" "446" "509" "520" "521" "562" "564" "565" "621" "647" "683" "762" "869" "890" "892" "894"\
    "922" "923" "927" "1007" "1029" "1048" "1051" "1069" "1096" "1126" "1172" "1180" "1184" "1188" "1187" "1200" "1201" "1209" "1239" "1242" "1248" "1257"\
    "1331" "1352" "1353" "267" "300" "383" "513" "516" "609" "613" "624" "686" "703" "785" "839" "925" "935" "945" "983" "985" "987" "1006" "1010" "1023" "1050" "1059"\
    "1072" "1074" "1077" "1078" "1082" "1097" "1105" "1139" "1140" "1148" "1185" "1186" "1197" "1198" "1203" "1205" "1232" "1235" "1236" "1238" "1243" "1256" "1258"\
    "1260" "1308" "1311" "1324" "1326" "1328" "1330" "1333" "1339" "1346" "1347" "1349" "264" "265" "290" "295" "552" "567" "783" "1014" "1068" "1136" "1202" "1247" \
    "1338" "1351")

for P in ${PATIENTS[@]}
do
    # echo $P
    # python3 parser.py $IMZML_ROI_ROOT_DIR/validation/"$P.imzML" $IMZML_ROI_EXTRACTED_ROOT_DIR/validation/$P

    if [ -f $IMZML_ROI_ROOT_DIR/validation/"$P.imzML" ] && [ -f $IMZML_ROI_ROOT_DIR/validation/"$P.ibd" ]; then
        # echo $P
        python3 parser.py $IMZML_ROI_ROOT_DIR/validation/"$P.imzML" $IMZML_ROI_EXTRACTED_ROOT_DIR/validation/$P
    fi

    # if [ ! -f $IMZML_ROI_ROOT_DIR/validation/"$P.imzML" ]; then
    #     echo "$P.imzML"
    # fi

    # if [ ! -f $IMZML_ROI_ROOT_DIR/validation/"$P.ibd" ]; then
    #     echo "$P.ibd"
    # fi
done

echo "----- Validation exvivo"

mkdir -p $IMZML_ROI_EXTRACTED_ROOT_DIR/validation_exvivo

# ROI
PATIENTS=("1076_exvivo" "1084_exvivo" "1147_exvivo" "1172_exvivo" "1188_exvivo" "1283_exvivo" "1294_exvivo" "1328_exvivo" "1331_exvivo" "250_exvivo" "290_exvivo"\
    "992_exvivo" "995_exvivo")

# WHOLE
PATIENTS_WHOLE=("I17_19488_a_exvivo" "I18_3561_exvivo" "I18_4057_exvivo" "5883_2_exvivo" "1005_exvivo" "1188_exvivo" "I19_10933_exvivo" "I19_10050_exvivo" "I19_10092_exvivo")


for P in ${PATIENTS[@]}
do
    # echo $P
    # python3 parser.py $IMZML_ROI_ROOT_DIR/validation_exvivo/"$P.imzML" $IMZML_ROI_EXTRACTED_ROOT_DIR/validation_exvivo/$P

    if [ -f $IMZML_ROI_ROOT_DIR/validation_exvivo/"$P.imzML" ] && [ -f $IMZML_ROI_ROOT_DIR/validation_exvivo/"$P.ibd" ]; then
        # echo $P
        python3 parser.py $IMZML_ROI_ROOT_DIR/validation_exvivo/"$P.imzML" $IMZML_ROI_EXTRACTED_ROOT_DIR/validation_exvivo/$P
    fi

    # if [ ! -f $IMZML_ROI_ROOT_DIR/validation_exvivo/"$P.imzML" ]; then
    #     echo "$P.imzML"
    # fi

    # if [ ! -f $IMZML_ROI_ROOT_DIR/validation_exvivo/"$P.ibd" ]; then
    #     echo "$P.ibd"
    # fi
done

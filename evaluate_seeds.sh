#!/bin/bash

for SEED in 1112 1117 1119 1121 1123 1124 1125 1129 1132 1133 1134 1137 1141 1142 1143 1144 1153 1162 1163 1167 1168 1169 1170 1174 1176 1179 1180 1182 1183 1184
do
    kedro run --params seed=$SEED
done
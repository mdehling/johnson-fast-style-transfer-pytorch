#!/bin/sh
destdir=${1:-"coco2014"}
url="http://images.cocodataset.org/zips/train2014.zip"
file="train2014.zip"

mkdir -p "${destdir}"
if [ $? != 0 ]; then
    echo "could not create ${destdir}"
    exit 1
fi

if [ ! -f "${destdir}/${file}" ]; then
    aria2c -x 10 -j 10 -d "${destdir}" -o "${file}" "${url}" || exit 1
fi

cd "${destdir}"
#7za x "${file}" && rm -f "${file}"
unzip "${file}" && rm -f "${file}"

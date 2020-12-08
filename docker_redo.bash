start=$(date +%s)

# CLEAN UP
docker ps -a | awk '{ print $1,$2 }' |  awk '{print $1 }' | xargs -I {} docker rm -f {}
docker images | grep simplequery | awk '{print $3 }' | xargs -I {} docker rmi  {}
docker images | grep maindb | awk '{print $3 }' | xargs -I {} docker rmi  {}
docker network ls | grep appNetwork | awk '{print $1 }' | xargs -I {} docker network rm  {}

# RESTART
sudo mount -o discard,defaults /dev/sdb /mnt/disks/ssd_disk
sudo chmod a+w /mnt/disks/ssd_disk
docker image build -t maindb -f maindb/Docker_db ./maindb
docker image build -t simplequery -f frontend/Docker_frontend ./frontend
docker network create appNetwork

# mount a persistent GCP volume
docker run --name simplequery -d -p 5000:8081 -e DB_HOST=maindb -v /mnt/disks/ssd_disk/final:/static --network appNetwork simplequery
docker run --name maindb -d -v /mnt/disks/ssd_disk/final:/static  --network appNetwork maindb
sleep 10 && docker ps -a && docker logs simplequery && docker logs maindb

end=$(date +%s)

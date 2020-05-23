hosts=$1
execu=$2
tp=$3
method=$4

read -r serv < $hosts
serv=($serv)
serv=${serv[0]}
# echo ${serv[0]}

tot=`wc -l < $hosts`
i=0
# echo "total hosts: " + $tot

python3 $execu $serv $tot $tp ${serv[0]} $method &
while read line
do
    # echo $i
    if (( $i != 0))
    then
        # echo "EXEC"
        arr=($line)
        ssh ${arr[1]}@${arr[0]} python3.7 -u - $serv $tot $tp ${arr[0]} $method < $execu &
    else
        ((++i))
    fi
done < $hosts

wait
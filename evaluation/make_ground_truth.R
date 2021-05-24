recording_id = "09"
recording_id = "22"
recording_id = "23"
recording_id = "35"
recording_id = "39"
recording_id = "46"
recording_id = "61"
recording_id = "64"
recording_id = "84"
recording_id = "91"
recording_id = "93"

dir.help = paste('d:\\dane\\kitti\\2011_09_26\\2011_09_26_drive_00', recording_id, '_sync\\oxts\\data\\', sep='')
file = '0000000000.txt'
file2 = paste(dir.help, file, sep = '')
df = read.csv(file2, header = FALSE, sep=' ')
names.help = c('lat', 'lon',  'alt',  'roll', 'pitch','yaw', 'vn','ve','vf', 'vl', 'vu', 'ax', 'ay', 'ay',
  'af',  'al',  'au',  'wx',  'wy',  'wz',  'wf',  'wl',  'wu',  'pos_accuracy',  'vel_accuracy', 'navstat',
  'numsats',  'posmode',  'velmode',  'orimode')

names(df) = names.help

lf = list.files(dir.help)

for (a in 2:length(lf))
{
  file2 = paste(dir.help, lf[a], sep = '')
  vv = read.csv(file2, header = FALSE, sep=' ')
  names(vv) = names.help
  df <- rbind(df, vv)
}

plot(df$lon, df$lat)
points(df$lon[395], df$lat[395], col='red')
dist.lon = 0
dist.lat = 0

dist.lon2 = 0
dist.lat2 = 0

library(geosphere)
for (a in 2:length(lf))
{
  #print(distm(c(df$lon[a - 1], df$lat[a - 1]), c(df$lon[a], df$lat[a]), fun = distHaversine)[1,1])
  dist.lon2 =c(dist.lon2, (distm(c(df$lon[a - 1], df$lat[a - 1]), c(df$lon[a], df$lat[a - 1]), fun = distHaversine)[1,1]))
  dist.lat2 =c(dist.lat2, (distm(c(df$lon[a - 1], df$lat[a - 1]), c(df$lon[a - 1], df$lat[a]), fun = distHaversine)[1,1]))
  
  dist.lon =c(dist.lon, (distm(c(df$lon[1], df$lat[1]), c(df$lon[a], df$lat[a - 1]), fun = distHaversine)[1,1]))
  dist.lat =c(dist.lat, (distm(c(df$lon[1], df$lat[1]), c(df$lon[a - 1], df$lat[a]), fun = distHaversine)[1,1]))
}
df.to.write = data.frame(distlon = dist.lon, distlat = dist.lat, distlon2 = dist.lon2, distlat2 = dist.lat2)
#write.csv(df.to.write, 'd:\\dane\\kitti\\2011_09_26\\2011_09_26_drive_0009_sync\\ground_truth.txt', row.names = FALSE, quote = FALSE)
write.csv(df.to.write, paste('d:\\dane\\kitti\\2011_09_26\\2011_09_26_drive_00', recording_id, '_sync\\ground_truth.txt', sep=''), row.names = FALSE, quote = FALSE)


#for (a in 2:length(lf))
#{
#  print(distm(c(df$lon[a - 1], df$lat[a - 1]), c(df$lon[a], df$lat[a]), fun = distHaversine)[1,1])
#}

plot(dist.lon, dist.lat)
plot(dist.lon)

a = 2
print(distm(c(df$lon[a - 1], df$lat[a - 1]), c(df$lon[a], df$lat[a]), fun = distHaversine)[1,1])
v1 =(distm(c(df$lon[a - 1], df$lat[a - 1]), c(df$lon[a], df$lat[a - 1]), fun = distHaversine)[1,1])
v2 =(distm(c(df$lon[a - 1], df$lat[a - 1]), c(df$lon[a - 1], df$lat[a]), fun = distHaversine)[1,1])
sqrt(v1 ^ 2 + v2 ^ 2)
#library(geosphere)
#distm(c(lon1, lat1), c(lon2, lat2), fun = distHaversine)
#write.csv(df, 'd:\\dane\\kitti\\2011_09_26\\2011_09_26_drive_0009_sync\\df09.txt', row.names = FALSE, quote = FALSE)
write.csv(df, paste('d:\\dane\\kitti\\2011_09_26\\2011_09_26_drive_00',recording_id ,'_sync\\df',recording_id, '.txt', sep=''), row.names = FALSE, quote = FALSE)
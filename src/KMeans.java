import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.*;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class KMeans extends Configured implements Tool {

    private static String DATA_FILE_NAME = "data.txt";
    private static String OUT = "outfile";
    private static String CENTROID_FILE_NAME = "c1.txt";
    private static String OUTPUT_FILE_NAME = "/part-00000";
    private static String JOB_NAME = "KMeans";
    private static ArrayList<ArrayList<Double>> kCentroids;
    private static double phiCost = 0.0d;
    private static String COST_FILE = "cost.txt";

    private static Double[] convertStrToDouble(String[] arr) {
        Double[] d = new Double[arr.length];
        for (int i = 0; i < arr.length; i++) {
            d[i] = Double.parseDouble(arr[i]);
        }
        return d;
    }

    public static class Map extends MapReduceBase implements Mapper<LongWritable, Text, Text, Text> {

        @Override
        public void close() throws IOException {
            FileWriter writer = new FileWriter(COST_FILE, true);
            writer.write(phiCost + "\n");
            writer.close();
        }

        @Override
        public void configure(JobConf job) {
            phiCost = 0;
            URI[] cachedFiles;
            try {
                cachedFiles = DistributedCache.getCacheFiles(job);
                kCentroids = new ArrayList<>();
                FileSystem fs = FileSystem.get(job);
                BufferedReader fileReader = new BufferedReader(new InputStreamReader(fs.open(new Path(cachedFiles[0].getPath()))));
                String line;
                while ((line = fileReader.readLine()) != null) {
                    line = line.trim();
                    String[] splitStr = line.split(" ");
                    Double[] centroid = convertStrToDouble(splitStr);
                    kCentroids.add(new ArrayList<>(Arrays.asList(centroid)));
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        public static double computeEudDistance(Double[] x, Double[] y) {
            double dis = 0;
            for (int i = 0; i < x.length; i++) {
                dis += (x[i] - y[i]) * (x[i] - y[i]);
            }
            return dis;
        }

        public static double computeManDistance(Double[] x, Double[] y) {
            double dis = 0;
            for (int i = 0; i < x.length; i++) {
                dis += Math.abs(x[i] - y[i]);
            }
            return dis;
        }

        @Override
        public void map(LongWritable key, Text value,
                        OutputCollector<Text, Text> output,
                        Reporter reporter) throws IOException {
            String[] pointStr = value.toString().trim().split("\\s");
            Double[] point = convertStrToDouble(pointStr);
            double minDis = Double.MAX_VALUE;
            int index = -1;
            for (int i = 0; i < kCentroids.size(); i++) {
                List<Double> currentCentroid = kCentroids.get(i);
                Double[] cen = currentCentroid.toArray(new Double[currentCentroid.size()]);
                //double eud = computeEudDistance(point, cen);
                double eud = computeManDistance(point, cen);
                if (eud < minDis) {
                    minDis = eud;
                    index = i;
                }
            }
            output.collect(new Text(kCentroids.get(index).toString()), new Text(value.toString()));
            phiCost += minDis;
        }
    }

    public static class Reduce extends MapReduceBase implements Reducer<Text, Text, Text, Text> {


        @Override
        public void reduce(Text text, Iterator<Text> iterator, OutputCollector<Text, Text> outputCollector, Reporter reporter) throws IOException {
            //Calculate new centroid
            double[] newCentroid = new double[58];
            int count = 0;
            while (iterator.hasNext()) {
                String[] line = iterator.next().toString().split(" ");
                Double[] point = convertStrToDouble(line);
                for (int i = 0; i < point.length; i++) {
                    newCentroid[i] += point[i];
                }
                count++;
            }
            for (int i = 0; i < newCentroid.length; i++) {
                newCentroid[i] /= count;
            }
            StringBuilder newCent = new StringBuilder();
            for (int i = 0; i < 58; i++) {
                newCent.append(newCentroid[i] + " ");
            }

            outputCollector.collect(new Text(newCent.toString()), new Text());
        }
    }

    public static void main(String[] args) throws Exception {
        ToolRunner.run(new Configuration(), new KMeans(), args);
    }

    public int run(String[] args) throws Exception {
        OUT = args[1];

        String output = OUT + System.nanoTime();
        String ROOT = output;

        for (int i = 0; i < 20; i++) {
            JobConf conf = new JobConf(KMeans.class);
            if (i == 0) {
                Path localPath = new Path(CENTROID_FILE_NAME);
                DistributedCache.addCacheFile(localPath.toUri(), conf);
            } else {
                Path localPath = new Path(ROOT + OUTPUT_FILE_NAME);
                DistributedCache.addCacheFile(localPath.toUri(), conf);
            }
            conf.setJobName(JOB_NAME);
            conf.setMapOutputKeyClass(Text.class);
            conf.setMapOutputValueClass(Text.class);
            conf.setOutputKeyClass(Text.class);
            conf.setOutputValueClass(Text.class);
            conf.setMapperClass(Map.class);
            conf.setReducerClass(Reduce.class);
            conf.setInputFormat(TextInputFormat.class);
            conf.setOutputFormat(TextOutputFormat.class);

            FileInputFormat.setInputPaths(conf, new Path(DATA_FILE_NAME));
            FileOutputFormat.setOutputPath(conf, new Path(output));

            JobClient.runJob(conf);
            ROOT = output;
            output = OUT + System.nanoTime();
        }
        return 0;

    }
}

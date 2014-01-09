package com.bah.externship;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
import java.util.StringTokenizer;

import org.apache.commons.math.linear.MatrixUtils;
import org.apache.commons.math.linear.RealMatrix;
import org.apache.commons.math.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class Portfolio extends Configured implements Tool{

	@Override
	public int run(String[] args) throws Exception {
		Job job = new Job(getConf(), "WordCount");
		job.setJarByClass(WordCount.class);

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);

		job.setMapperClass(Map.class);
		job.setCombinerClass(Reduce.class);
		job.setReducerClass(Reduce.class);

		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);

		TextInputFormat.setInputPaths(job, new Path("/user/hue/jobsub/sample_data/midsummer.txt"));
		TextOutputFormat.setOutputPath(job, new Path("/user/hue/jobs/" + System.currentTimeMillis()));

		job.waitForCompletion(true);
		return 0;
	}

	public static class Map extends Mapper<LongWritable, Text, Text, IntWritable> {
		private final static IntWritable one = new IntWritable(1);
		private Text word = new Text();

		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			String line = value.toString();
			StringTokenizer tokenizer = new StringTokenizer(line);
			while (tokenizer.hasMoreTokens()) {
				word.set(tokenizer.nextToken()); // general practice
				context.write(word, one);
			}
		}
	}

	public ArrayList<String[]> parse()	{
		File file=new File("docs/NYSE-2000-2001.tsv/NYSE-2000-2001.tsv");
		double last=0;
		double curr=0;
		String lastname="";
		String currname="";
		ArrayList<String[]> data=null;
		try {
			Scanner reader=new Scanner(file);
			data=new ArrayList<String[]>();
			reader.nextLine();
			while(reader.hasNextLine()){
				String s=reader.nextLine().trim();
				String[] in=s.split("\\s");
				currname=in[1];
				System.out.println(currname);
				if(currname.compareTo(lastname)!=0)
					last=0;

				curr=Double.parseDouble(in[6]);
				if(last!=0)	{
					String[] str={in[1],in[2]+"r_"+Math.log(curr/last)};
					data.add(str);
					System.out.println(Arrays.asList(str));
				}
				last=curr;
				lastname=currname;
				
				reader.close();
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		return data;
	}

	/**
	 * @author MultiLaptopSystem
	 *
	 */
	public static class Reduce extends Reducer<Text, IntWritable, Text, IntWritable> {
		private IntWritable count = new IntWritable();
		private static final int position = 1000000;
        private static final double alpha = new NormalDistribution().inverseCumulativeProbability(0.95);
        
		public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
			int sum = 0;
			for(IntWritable value : values) {
				sum += value.get();
			}
			count.set(sum);
			context.write(key, count);
		}
		
		/**
		 * 
		 * @param al1 ArrayList 1 of ln returns of a company.
		 * @param al2 ArrayList 2 of ln returns of another company.
		 * Method calculates correlation using PearsonsCorrelation() in Apache Math.
		 * Arraylists must each be converted to array of doubles to work with PearsonsCorrelation().correlation().
		 * @return double of correlation between company 1 and 2.
		 */
		public double getCorrelation(ArrayList<Integer> al1, ArrayList<Integer> al2){
			double [] company1 = new double[al1.size()];
			double [] company2 = new double[al2.size()];
			int counter = 0;
			for (int i : al1){
				company1[counter] = i;
			}
			counter = 0;
			for (int i : al2){
				company2[counter] = i;
			}
			double z = new PearsonsCorrelation().correlation(company1, company2);
			return z;
		}
        
        /**
         * Return the VaR (Value at Risk) using volatility which is the standard deviation of a company.
         * @alpha remains constant: the "NORMSINV" of a 95% confidence interval.
         * @position remains constant: the number of shares held.
         * @return The number of objects contained
         */
        public double valueAtRisk(int volatility){
        	return alpha * position * volatility;
        }
    	double [][] CorrelationSquare = new double[29][29];
    	
    	/**
    	 * Sami
    	 * Fills the correlation square when given correlation values used after calling getCorrelation.
    	 * @param companyKey1
    	 * @param companyKey2
    	 * @param correlation
    	 */
    	public void fillCorrelationSquare(int companyKey1, int companyKey2, double correlation){
    		this.CorrelationSquare[companyKey1][companyKey2]=correlation;
    		this.CorrelationSquare[companyKey2][companyKey1]=correlation;
    	}
    	
    	/**
    	 * Sami
    	 * First matrix multiplication of transposed VaRs and correlation square.
    	 * @param valuesAtRisk
    	 * @return another real matrix (to be used with varStepTwo.
    	 */
    	public RealMatrix varStepOne(ArrayList<Integer>valuesAtRisk){
    		double [] transposedValuesAtRisk = new double [29];
    		int counter = 0;
    		for (int i : valuesAtRisk){
    			transposedValuesAtRisk[counter] = i;
    			counter ++;
    		}
    		RealMatrix correlationSquareMatrix = MatrixUtils.createRealMatrix(CorrelationSquare);
    		RealMatrix transposeSquareMatrix = MatrixUtils.createRowRealMatrix(transposedValuesAtRisk);
    		return transposeSquareMatrix.multiply(correlationSquareMatrix);
    	}
	}

	public static void main(String[] args) throws Exception {

		int res = ToolRunner.run(new Configuration(), new WordCount(), args);
		System.exit(res);
	}
}

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;

import scala.Tuple2;

public class RecommendationEngine {

  public static void main(String[] args) {

    // Turn off unnecessary logging
    Logger.getLogger("org").setLevel(Level.OFF);
    Logger.getLogger("akka").setLevel(Level.OFF);

    // Create Java spark context
    SparkConf conf = new SparkConf().setAppName("recoEngine").setMaster("local[2]");

    JavaSparkContext sc = new JavaSparkContext(conf);

    // Read user-item rating file. format - userId,movieId,rating,timestamp
    JavaRDD<String> userItemRatingsFile = sc.textFile("src/main/resources/user_ratings.csv");

    // Read item description file. format - movieId, movieName, Genre
    JavaRDD<String> itemDescriptionFile = sc.textFile("src/main/resources/movies_details.csv");

    // Map file to Ratings(user,item,rating) tuples
    JavaRDD<Rating> ratings = userItemRatingsFile.map(RecommendationEngine::getRatingsTuple);

    // Create tuples(itemId,ItemDescription), will be used later to get names of item from itemId
    JavaPairRDD<Integer, String> itemIdDescritpionPairRdd = itemDescriptionFile.mapToPair(getItemIdDescriptionTuple());

    // Build the recommendation model using ALS

    int rank = 10; // 10 latent factors
    int numIterations = 10; // number of iterations

    //ALS.trainImplicit(arg0, arg1, arg2)
    MatrixFactorizationModel model = ALS.trainImplicit(JavaRDD.toRDD(ratings), rank, numIterations);

    // Create userId-itemId tuples from ratings
    JavaPairRDD<Integer, Integer> userProducts = ratings.mapToPair(getUserIdMovieIdTuple());

    // Calculate the itemIds not rated by a particular user, say user with userId = 0
    JavaPairRDD<Integer, Integer> itemsNotRatedByUser = userProducts.filter(userProduct -> userProduct._1 != 0);

    // Predict the ratings of the items not rated by user for the user
    JavaRDD<Rating> recommendations = model.predict(itemsNotRatedByUser).distinct();

    // Sort the recommendations by rating in descending order
    recommendations = recommendations.sortBy((Function<Rating, Double>) userRating -> userRating.rating(), false, 1);

    // Get top 10 recommendations
    JavaRDD<Rating> topRecommendations = sc.parallelize(recommendations.take(10));

    // Join top 10 recommendations with item descriptions

    JavaRDD<Tuple2<Rating, String>> recommendedItems = topRecommendations.mapToPair(getMovieIdRatingTuple())
        .join(itemIdDescritpionPairRdd).values();

    //Print the top recommendations for user 0.
    System.out.println("Movie Id " + "\t\t" + "rating" + "\t\t\t\t" + "Movie Name");
    recommendedItems
        .foreach(result -> System.out.println(result._1.product() + "\t\t" + result._1.rating() + "\t\t" + result._2));

  }

  private static PairFunction<Rating, Integer, Rating> getMovieIdRatingTuple() {
    return (PairFunction<Rating, Integer, Rating>) input -> new Tuple2<>(input.product(), input);
  }

  private static PairFunction<String, Integer, String> getItemIdDescriptionTuple() {
    return (PairFunction<String, Integer, String>) line -> new Tuple2<>(Integer.parseInt(line.split(",")[0]),
        line.split(",")[1]);
  }

  private static Rating getRatingsTuple(String s) {
    String[] strArray = s.split(",");
    return new Rating(Integer.parseInt(strArray[0]), Integer.parseInt(strArray[1]), Double.parseDouble(strArray[2]));
  }

  private static PairFunction<Rating, Integer, Integer> getUserIdMovieIdTuple() {
    return (PairFunction<Rating, Integer, Integer>) inp -> new Tuple2<>(inp.user(), inp.product());
  }

}
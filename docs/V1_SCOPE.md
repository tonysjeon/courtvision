# TennisVision V1 Implementation Scope

## 1. Project Goal

Build an end-to-end tennis computer vision and data engineering platform that converts recorded tennis match footage into structured player and ball tracking data, tactical metrics, and interactive visualizations.

The system should:

1. Ingest a recorded singles tennis match video
2. Detect the tennis court and calculate a court-to-image transformation
3. Detect and track both players
4. Detect and track the tennis ball where possible
5. Map player and ball positions into normalized court coordinates
6. Generate structured tracking events
7. Stream events through Kafka
8. Process events using Spark / Databricks
9. Store transformed datasets using a bronze-silver-gold architecture
10. Compute player, rally, and court-position metrics
11. Load analytics-ready data into Snowflake
12. Transform warehouse data with dbt
13. Expose results through FastAPI
14. Display match analytics through a Next.js dashboard

The primary V1 success criterion is:

> Given a tennis match clip from a standard broadcast camera, reconstruct the court, track both players, map their movement into normalized court coordinates, and produce queryable tactical analytics through an end-to-end data pipeline.

---

# 2. V1 Scope

V1 should support:

* Singles tennis only
* Recorded video only
* Standard broadcast-style camera angles
* One continuous point, game, or short match segment
* Court calibration
* Player detection
* Player tracking
* Basic ball tracking
* Normalized court mapping
* Tracking event generation
* Data engineering pipeline
* Tactical metrics
* Match dashboard

Reliability and clean architecture are more important than supporting every tennis scenario.

---

# 3. V1 Non-Goals

Do not implement the following in V1:

* Live broadcast ingestion
* Doubles
* Player facial recognition
* Automatic player-name identification
* Full Hawk-Eye-level ball accuracy
* Exact ball speed
* Spin estimation
* Pose estimation
* Forehand/backhand recognition
* Shot-type classification
* Serve type classification
* Automated line calling
* Advanced tactical recommendations
* Match outcome prediction
* Authentication
* Mobile application
* Kubernetes
* Production-scale distributed infrastructure

These can be considered future phases.

---

# 4. Technology Stack

## Computer Vision

* Python
* PyTorch
* OpenCV
* Ultralytics YOLO
* ByteTrack or equivalent tracking library

Potential later additions:

* TrackNet-style ball tracking
* ONNX
* TensorRT
* C++ inference

Do not add these until the Python V1 works.

## Streaming

* Apache Kafka
* Docker Compose
* Kafka UI

## Data Engineering

* PySpark
* Databricks
* Delta Lake

## Analytics Warehouse

* Snowflake
* dbt Core

## Backend

* Python
* FastAPI
* Pydantic

## Frontend

* Next.js
* TypeScript
* D3.js
* SVG or Canvas for court rendering

## Developer Tooling

* Docker
* Docker Compose
* pytest
* Ruff
* ESLint
* Prettier
* GitHub Actions

---

# 5. High-Level Architecture

```text
Tennis Match Video
        |
        v
Video Ingestion
        |
        v
Court Detection
        |
        v
Homography / Court Calibration
        |
        +------------------------+
        |                        |
        v                        v
Player Detection            Ball Detection
        |                        |
        v                        v
Player Tracking             Ball Tracking
        |                        |
        +------------+-----------+
                     |
                     v
         Normalized Court Coordinates
                     |
                     v
             Tracking Events
                     |
                     v
                   Kafka
                     |
                     v
             Databricks / Spark
                     |
          +----------+----------+
          |          |          |
          v          v          v
       Bronze      Silver      Gold
                                |
                                v
                         Snowflake / dbt
                                |
                                v
                            FastAPI
                                |
                                v
                     Next.js + D3 Dashboard
```

---

# 6. Repository Structure

Use a monorepo.

```text
tennisvision/
├── apps/
│   ├── api/
│   │   ├── app/
│   │   └── tests/
│   │
│   └── web/
│       ├── app/
│       ├── components/
│       └── lib/
│
├── services/
│   ├── vision/
│   │   ├── video/
│   │   ├── court/
│   │   ├── players/
│   │   ├── ball/
│   │   ├── tracking/
│   │   ├── geometry/
│   │   └── visualization/
│   │
│   └── event_producer/
│
├── pipelines/
│   ├── spark/
│   │   ├── bronze/
│   │   ├── silver/
│   │   ├── gold/
│   │   └── tests/
│   │
│   └── dbt/
│
├── data/
│   ├── sample/
│   ├── processed/
│   └── schemas/
│
├── infrastructure/
│   ├── kafka/
│   └── docker/
│
├── docs/
│   ├── architecture.md
│   ├── data-model.md
│   ├── cv-pipeline.md
│   └── V1_SCOPE.md
│
├── tests/
├── docker-compose.yml
├── Makefile
├── .env.example
└── README.md
```

The vision system must be runnable independently of Kafka, Databricks, Snowflake, and the frontend.

---

# 7. Phase 1: Video Ingestion

Create a reusable video ingestion layer.

Input:

```text
.mp4
.mov
```

Extract:

```text
fps
frame_count
width
height
duration_seconds
```

Each frame should contain:

```text
frame_id
timestamp_ms
image
```

Expose an interface similar to:

```python
reader = VideoReader(path)

for frame in reader:
    pipeline.process(frame)
```

Support:

* frame skipping
* maximum frame count
* start timestamp
* end timestamp

This makes development on short clips much faster.

---

# 8. Phase 2: Court Detection

The system needs to identify the visible tennis court.

A tennis court provides strong geometric constraints because the dimensions and line layout are standardized.

Detect or identify important court landmarks such as:

```text
near baseline left
near baseline right
far baseline left
far baseline right

near service line left
near service line right
far service line left
far service line right

center service line

sidelines
baseline intersections
service box intersections
```

## Initial V1 Approach

Start with manually selected court keypoints for one sample clip.

Store calibration points in a JSON file.

Example:

```json
{
  "near_left_baseline": [155, 692],
  "near_right_baseline": [1065, 692],
  "far_left_baseline": [415, 242],
  "far_right_baseline": [809, 242]
}
```

This allows the rest of the pipeline to be developed without blocking on automatic court detection.

## V1.1 Improvement

Implement automatic court-line detection using:

* OpenCV edge detection
* line detection
* court geometry constraints

or a dedicated court-keypoint model.

The architecture must allow manual and automatic calibration implementations to share the same interface.

Example:

```python
CourtDetector.detect(frame) -> CourtKeypoints
```

---

# 9. Phase 3: Court Homography

Create a normalized tennis court coordinate system.

Use real court dimensions conceptually, but normalize coordinates for V1.

Recommended coordinate system:

```text
x = 0 to 100
y = 0 to 100
```

Where:

```text
(0,0)    = near-left court corner
(100,0)  = near-right court corner
(0,100)  = far-left court corner
(100,100)= far-right court corner
```

Calculate a homography matrix using OpenCV.

Expose:

```python
court_x, court_y = court_mapper.transform(pixel_x, pixel_y)
```

The transformation should work for:

* player ground positions
* ball positions
* court landmarks

Write unit tests using known source/destination points.

---

# 10. Phase 4: Player Detection

Detect the two tennis players.

Start with a pretrained YOLO model using the `person` class.

For every player detection output:

```text
frame_id
timestamp_ms
confidence
bbox_x1
bbox_y1
bbox_x2
bbox_y2
center_x
center_y
ground_x
ground_y
```

Use the bottom-center of the bounding box as the player's ground position.

Do not assume every detected person is a player.

Spectators, line judges, ball kids, and officials may appear.

---

# 11. Phase 5: Player Filtering

Determine which detected people represent the active near-side and far-side players.

Useful heuristics:

* must be within or near the mapped playing area
* near-side player generally occupies the lower court region
* far-side player generally occupies the upper court region
* player detections should persist across frames
* prefer detections near known previous player location

Represent:

```text
player_id = "near_player"
player_id = "far_player"
```

Do not attempt actual player identity in V1.

---

# 12. Phase 6: Player Tracking

Use ByteTrack or another multi-object tracker.

Maintain persistent tracks for:

```text
near_player
far_player
```

Output:

```text
match_id
frame_id
timestamp_ms
player_id
track_id
confidence
pixel_x
pixel_y
court_x
court_y
```

V1 success requirement:

> Both players retain stable identities during a continuous rally under normal broadcast conditions.

Track switches should be logged.

---

# 13. Phase 7: 2D Court Reconstruction

Create a side-by-side development visualization:

```text
ORIGINAL VIDEO             NORMALIZED COURT

     far player                  ●


      tennis court        -----------------
                          |               |
                          |       ●       |
                          |---------------|
                          |               |
                          |    ●          |
                          |               |
                          -----------------

     near player
```

Render:

* near player
* far player
* court boundaries
* service boxes
* current timestamp
* player track IDs

This should update frame by frame.

V1 milestone is complete when the dots on the normalized court visually correspond to player movement in the video.

---

# 14. Phase 8: Ball Detection

Ball tracking is expected to be the most difficult CV portion.

Start simple.

## Initial Approach

Attempt ball detection using:

* YOLO
* motion filtering
* small-object detection heuristics

Output candidate:

```text
frame_id
timestamp_ms
confidence
pixel_x
pixel_y
```

Do not require perfect ball detection for the first end-to-end milestone.

The rest of the pipeline must operate when ball observations are missing.

---

# 15. Phase 9: Ball Tracking

Create a temporal ball tracker.

The tracker should use:

* previous ball location
* velocity estimate
* candidate proximity
* motion direction
* short-gap interpolation

Output:

```text
match_id
frame_id
timestamp_ms
object_type = "ball"
pixel_x
pixel_y
court_x
court_y
confidence
is_interpolated
```

For invalid or uncertain ball observations, preserve nulls rather than fabricating high-confidence positions.

---

# 16. Phase 10: Ball Bounce Detection

This is optional for the first V1 demo but should be attempted after stable ball tracking.

Estimate potential bounce events based on trajectory changes.

Event:

```json
{
  "event_type": "ball_bounce",
  "timestamp_ms": 19250,
  "court_x": 62.4,
  "court_y": 71.2,
  "confidence": 0.76
}
```

Avoid claiming line-call accuracy.

Bounce detection is analytical, not officiating-grade.

---

# 17. Data Event Schema

Define stable Pydantic schemas.

## Player Tracking Event

```json
{
  "schema_version": "1.0",
  "match_id": "demo_match_001",
  "frame_id": 248,
  "timestamp_ms": 8266,
  "object_type": "player",
  "object_id": "near_player",
  "track_id": 4,
  "confidence": 0.96,
  "pixel_position": {
    "x": 644.2,
    "y": 698.1
  },
  "court_position": {
    "x": 54.3,
    "y": 12.8
  }
}
```

## Ball Tracking Event

```json
{
  "schema_version": "1.0",
  "match_id": "demo_match_001",
  "frame_id": 248,
  "timestamp_ms": 8266,
  "object_type": "ball",
  "object_id": "ball",
  "track_id": null,
  "confidence": 0.73,
  "pixel_position": {
    "x": 612.7,
    "y": 401.3
  },
  "court_position": {
    "x": 48.9,
    "y": 56.2
  },
  "is_interpolated": false
}
```

---

# 18. Phase 11: Kafka Event Streaming

Create a Kafka producer.

Initial topic:

```text
tennis-tracking-events
```

Optional topics:

```text
tennis-match-events
pipeline-metrics
```

Kafka producer requirements:

* JSON serialization
* Pydantic validation
* batching
* retries
* logging
* graceful shutdown

Docker Compose should run:

```text
Kafka
Kafka UI
```

The vision pipeline should support two output modes:

```text
--output file
--output kafka
```

This allows development without Kafka running.

---

# 19. Phase 12: Databricks / Spark Bronze Layer

Consume tracking events.

Store raw records with minimal transformation.

Table:

```text
bronze_tracking_events
```

Columns:

```text
schema_version
match_id
frame_id
timestamp_ms
object_type
object_id
track_id
confidence
pixel_x
pixel_y
court_x
court_y
is_interpolated
ingested_at
```

Bronze should preserve original observations.

---

# 20. Phase 13: Silver Player Tracking

Create:

```text
silver_player_tracking
```

Perform:

* null validation
* coordinate-bound filtering
* duplicate removal
* timestamp ordering
* short-gap interpolation
* movement calculations

Derived columns:

```text
delta_time
delta_x
delta_y
distance_delta
velocity
```

Example:

```text
match_id
player_id
timestamp_ms
court_x
court_y
distance_delta
velocity
```

---

# 21. Phase 14: Silver Ball Tracking

Create:

```text
silver_ball_tracking
```

Transform:

* remove impossible coordinates
* mark missing frames
* interpolate short gaps
* calculate trajectory velocity
* retain confidence
* retain interpolation flag

Example:

```text
match_id
timestamp_ms
court_x
court_y
velocity_x
velocity_y
speed
confidence
is_interpolated
```

---

# 22. Phase 15: Rally Segmentation

Create basic rally segmentation.

V1 may initially use manual point boundaries.

Store:

```text
rally_id
start_timestamp_ms
end_timestamp_ms
```

Later automatically infer point boundaries using:

* player inactivity
* ball disappearance
* serve positioning
* scoreboard changes

Do not block V1 on automatic rally detection.

---

# 23. Phase 16: Gold Analytics Tables

Create analytics-ready datasets.

## Gold Player Match Metrics

```text
gold_player_match_metrics
```

Metrics:

### Distance Covered

Calculate cumulative movement:

```text
sum(distance_delta)
```

### Average Court Position

```text
avg(court_x)
avg(court_y)
```

### Lateral Movement

Calculate cumulative horizontal movement:

```text
sum(abs(delta_x))
```

### Depth Position

Average distance relative to baseline.

### Baseline Positioning

Percentage of time:

```text
behind_baseline
near_baseline
inside_baseline
```

### Court Coverage

Estimate area occupied using either:

* position grid coverage
* convex hull
* heatmap occupancy

---

# 24. Gold Rally Metrics

Create:

```text
gold_rally_metrics
```

Initial fields:

```text
match_id
rally_id
duration_seconds
near_player_distance
far_player_distance
near_player_avg_x
near_player_avg_y
far_player_avg_x
far_player_avg_y
ball_observation_count
```

If ball tracking supports it:

```text
bounce_count
average_ball_speed_proxy
```

Do not claim physically accurate ball speed unless proper camera calibration supports it.

---

# 25. Tactical Metrics

V1 should expose at least five useful tennis metrics.

## 1. Player Average Court Position

Show average location as a point on court.

## 2. Court Position Heatmap

Bin player positions into a grid.

Example:

```text
10 x 20 cells
```

Calculate occupancy frequency.

## 3. Distance Covered

Approximate total movement.

## 4. Baseline Depth

Measure how aggressively each player positions themselves.

Example:

```text
average meters / normalized units behind baseline
```

## 5. Lateral Movement

Measure side-to-side movement.

Optional:

## 6. Recovery Position

Measure where a player returns after shots or ball events.

## 7. Rally Movement Load

Distance covered during each rally.

---

# 26. Phase 17: Snowflake

Load gold tables into Snowflake.

Suggested tables:

```text
matches
players
player_match_metrics
rally_metrics
player_position_bins
```

Snowflake should contain analytics-ready data rather than every raw video-frame observation.

---

# 27. Phase 18: dbt

Use dbt Core to build warehouse models.

Example layers:

```text
staging
intermediate
marts
```

Example models:

```text
stg_player_match_metrics
stg_rally_metrics

int_player_position_profile

mart_match_summary
mart_player_tactical_profile
```

Add dbt tests:

```text
not_null
unique
relationships
accepted_values
```

---

# 28. Phase 19: FastAPI Backend

Build endpoints:

```text
GET /health

GET /matches

GET /matches/{match_id}

GET /matches/{match_id}/tracking

GET /matches/{match_id}/players

GET /matches/{match_id}/players/{player_id}

GET /matches/{match_id}/rallies

GET /matches/{match_id}/metrics

GET /matches/{match_id}/heatmaps
```

Example response:

```json
{
  "match_id": "demo_match_001",
  "players": {
    "near_player": {
      "distance_covered": 817.4,
      "average_position": {
        "x": 51.4,
        "y": 14.7
      },
      "baseline_distribution": {
        "behind": 0.53,
        "near": 0.32,
        "inside": 0.15
      }
    }
  }
}
```

---

# 29. Phase 20: Next.js Dashboard

Create:

```text
/matches/[matchId]
```

The page should include:

## Match Video

Display source or processed match video.

## Live Court Reconstruction

Draw a tennis court using SVG.

Overlay:

```text
near player
far player
ball
```

Allow timeline playback.

## Player Heatmaps

Display positional heatmap for each player.

## Tactical Metrics

Show:

```text
distance covered
average court depth
lateral movement
baseline positioning
court coverage
```

## Rally Table

Example:

```text
Rally | Duration | Near Distance | Far Distance
1     | 8.2s     | 14.3         | 12.8
2     | 16.1s    | 27.2         | 31.4
```

---

# 30. Synchronization Model

Tracking observations should be synchronized using:

```text
timestamp_ms
```

Frontend playback should:

1. Read current video timestamp
2. Find nearest tracking observation
3. Render player / ball positions
4. Update court visualization

Do not synchronize based only on frontend frame count.

---

# 31. Performance Instrumentation

Instrument the CV pipeline.

Track:

```text
video FPS
player detection latency
tracking latency
court transformation latency
ball detection latency
total frame latency
Kafka publish latency
```

Example benchmark:

```text
Source video FPS:          30
Processing FPS:            28.2
Player detection P50:      18 ms
Player detection P95:      24 ms
Tracking P95:               3 ms
Geometry P95:               1 ms
Total processing P95:      32 ms
```

Do not optimize prematurely.

First make the pipeline correct.

---

# 32. Logging

Use structured logging.

Example:

```json
{
  "timestamp": "...",
  "level": "INFO",
  "service": "vision",
  "match_id": "demo_match_001",
  "frame_id": 224,
  "message": "player tracking complete"
}
```

Log:

* missing court calibration
* invalid homography
* player track loss
* player track switch
* ball detection loss
* invalid court coordinate
* Kafka publish failure
* dropped frame

---

# 33. Testing

## Video Tests

* video metadata
* frame iteration
* timestamp calculations

## Court Tests

* homography transform
* court coordinate boundaries
* inverse transform if implemented

## Player Tests

* player event schema
* player filtering
* near/far assignment

## Data Pipeline Tests

* deduplication
* distance calculations
* movement calculations
* heatmap binning

## API Tests

* health
* matches
* tracking
* metrics
* heatmaps

---

# 34. Makefile

Provide commands such as:

```bash
make install
make test
make lint

make kafka-up
make kafka-down

make process-video VIDEO=data/sample/sample.mp4

make process-video-file VIDEO=data/sample/sample.mp4

make process-video-kafka VIDEO=data/sample/sample.mp4

make api
make web

make spark-bronze
make spark-silver
make spark-gold
```

---

# 35. Development Milestones

## Milestone 1: Video + Court

Goal:

```text
video
  ->
court calibration
  ->
normalized court mapping
```

Definition of done:

A manually calibrated sample clip correctly maps court coordinates into a normalized tennis court.

---

## Milestone 2: Player Tracking

Goal:

```text
video
  ->
player detection
  ->
near/far player identification
  ->
tracking
  ->
court coordinates
```

Definition of done:

Both players appear as synchronized moving points on the normalized court.

This is the first major demo milestone.

---

## Milestone 3: Ball Tracking

Goal:

```text
video
  ->
ball detections
  ->
temporal ball trajectory
  ->
court coordinates
```

Definition of done:

The system produces usable ball observations for a meaningful percentage of frames in a rally.

Do not block later infrastructure work if ball tracking remains imperfect.

---

## Milestone 4: Event Platform

Goal:

```text
CV observations
      ->
Pydantic events
      ->
Kafka
```

Definition of done:

Player and ball tracking events can be reliably published and consumed.

---

## Milestone 5: Databricks

Goal:

```text
Kafka / event files
      ->
Bronze
      ->
Silver
      ->
Gold
```

Definition of done:

Player movement metrics can be computed from tracking observations.

---

## Milestone 6: Analytics

Implement:

```text
distance covered
average court position
baseline positioning
lateral movement
court coverage
```

Definition of done:

Metrics are reproducible from gold-layer data.

---

## Milestone 7: Warehouse

Goal:

```text
Gold
 ->
Snowflake
 ->
dbt
```

Definition of done:

Analytics-ready match tables can be queried through Snowflake.

---

## Milestone 8: Product

Goal:

```text
FastAPI
   ->
Next.js
   ->
interactive match analysis
```

Definition of done:

A user can open one match and view:

* source video
* synchronized court reconstruction
* player movement
* heatmaps
* tactical metrics

---

# 36. Recommended Implementation Order

Do not build horizontally.

Use vertical slices.

Start with:

```text
one frame
   ->
court coordinates
```

Then:

```text
one player
   ->
court position
```

Then:

```text
two players
   ->
court visualization
```

Then:

```text
one event
   ->
Kafka
```

Then:

```text
one Kafka event
   ->
Spark
```

Then:

```text
one Spark metric
   ->
API
```

Then:

```text
one API result
   ->
frontend
```

Only after the entire vertical pipeline works should you expand volume and features.

---

# 37. Suggested First 10 Days

## Day 1

* initialize monorepo
* set up Python environment
* add video reader
* add sample clip
* generate frame metadata

## Day 2

* manually calibrate tennis court
* implement homography
* render normalized court

## Day 3

* add YOLO person detection
* filter active players
* map player locations to court

## Day 4

* integrate ByteTrack
* stabilize near/far player identity
* create synchronized court visualization

At this point, the first meaningful demo should work.

## Day 5

* build ball-detection prototype
* evaluate detection quality
* add temporal filtering

## Day 6

* create stable Pydantic event schemas
* persist events as JSON / Parquet
* add pipeline performance metrics

## Day 7

* add Kafka
* publish player / ball events
* add Kafka UI

## Day 8

* implement local PySpark bronze and silver layers
* calculate player movement

## Day 9

* add Databricks workflow
* implement gold analytics metrics
* generate heatmap datasets

## Day 10

* create FastAPI
* create initial Next.js match page
* visualize court positions and metrics

---

# 38. Definition of Done

V1 is complete when this command:

```bash
make process-video VIDEO=data/sample/sample.mp4
```

can produce a dataset containing player movement in normalized court coordinates.

The complete system should then support:

```text
Tennis Video
     ↓
Court Calibration
     ↓
Player Tracking
     ↓
Ball Tracking
     ↓
Spatial Event Generation
     ↓
Kafka
     ↓
Spark / Databricks
     ↓
Delta Bronze / Silver / Gold
     ↓
Snowflake / dbt
     ↓
FastAPI
     ↓
Next.js Dashboard
```

The final demo must visibly demonstrate:

1. original tennis footage
2. player detection
3. player tracking
4. normalized 2D court reconstruction
5. player movement heatmaps
6. tactical metrics
7. data pipeline architecture
8. queryable analytics

---

# 39. V1 Engineering Principle

Do not optimize the project around adding technologies to a resume.

Every technology should solve a clear problem:

```text
OpenCV / YOLO
    computer vision

ByteTrack
    temporal identity

Homography
    spatial normalization

Kafka
    event transport

Spark / Databricks
    high-volume event transformation

Delta Lake
    structured data layers

Snowflake
    analytical warehouse

dbt
    analytics modeling

FastAPI
    serving

Next.js / D3
    product visualization
```

Prioritize correctness, observability, modularity, and measurable performance.

A smaller system that works end-to-end is preferable to a larger system with disconnected components.

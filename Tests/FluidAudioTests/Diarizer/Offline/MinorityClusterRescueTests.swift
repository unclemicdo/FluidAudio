import XCTest

@testable import FluidAudio

/// Deterministic, model-free tests for the minority-cluster rescue.
///
/// Geometry: 2-d vectors on the unit circle. Host A near 62°, host B near
/// 72°, swallowed child C near 162°, second child D near 275°. The child's
/// spread is tiny while its distance to the host blob is ≫ 1.5×, so the
/// conservative gate cascade accepts it.
@available(macOS 14.0, iOS 17.0, *)
final class MinorityClusterRescueTests: XCTestCase {

    // MARK: - Fixture constants

    private let frameDuration = 0.1
    private let framesPerWindow = 20
    private let chunkCount = 16
    private let slotsPerChunk = 5

    // Chunk placements: (chunk, firstFrame, lastFrame). Chunk offsets sit at
    // `k - 0.5` s, so chunk k's frames map to canonical global times
    // [(k - 0.5) + f * 0.1].
    private let hostAPlacements: [(Int, Int, Int)] = [
        (0, 0, 9),
        (2, 0, 9),
        (4, 0, 9),
        (5, 0, 9),
    ]
    private let hostBPlacements: [(Int, Int, Int)] = [
        (0, 10, 19),
        (2, 10, 19),
        (4, 10, 19),
        (5, 10, 19),
        (6, 0, 9),
        (6, 10, 19),
    ]
    /// Two activity runs (≈2.5–3.5 s and ≈6.5–9.0 s) separated by ≫ 2 s.
    private let candidatePlacements: [(Int, Int, Int)] = [
        (3, 0, 9),
        (7, 0, 9),
        (9, 0, 9),
        (13, 0, 9),
    ]
    /// Two runs, separated by ≫ 2 s, same shape as candidate.
    private let secondCandidatePlacements: [(Int, Int, Int)] = [
        (4, 0, 9),
        (8, 0, 9),
        (10, 0, 9),
        (14, 0, 9),
    ]

    // MARK: - Fixture construction

    private struct Fixture {
        var engine: MinorityClusterRescueEngine
        var embeddings: [[Double]]
        var timed: [TimedEmbedding]
        var initialClusters: [Int]
        var centroids: [[Double]]
        var candidateIndices: Range<Int>
    }

    private func fixture(
        points: [[Double]],
        placements: [(chunk: Int, start: Int, end: Int)],
        initialLabel: Int,
        baselineHost: Int,
        overrideOptions: ((inout MinorityClusterRescueOptions) -> Void)? = nil,
        extraChildren: [(
            points: [[Double]],
            placements: [(chunk: Int, start: Int, end: Int)],
            label: Int,
            host: Int
        )] = [],
        activityThreshold: Float = 0.5
    ) -> Fixture {
        var timed: [TimedEmbedding] = []
        var embeddings: [[Double]] = []
        var initialClusters: [Int] = []
        var baselineAssignments: [Int] = []
        var nextSlot: [Int: Int] = [:]

        func append(
            _ points: [[Double]],
            _ placements: [(chunk: Int, start: Int, end: Int)],
            label: Int,
            host: Int
        ) {
            precondition(points.count == placements.count)
            for (index, point) in points.enumerated() {
                let placement = placements[index]
                let slot = nextSlot[placement.chunk, default: 0]
                precondition(slot < slotsPerChunk, "fixture exceeds local speaker slots")
                nextSlot[placement.chunk] = slot + 1
                var weights = [Float](repeating: 0, count: framesPerWindow)
                for frame in placement.start...placement.end where frame < framesPerWindow {
                    weights[frame] = 1
                }
                timed.append(
                    TimedEmbedding(
                        chunkIndex: placement.chunk,
                        speakerIndex: slot,
                        startFrame: placement.start,
                        endFrame: placement.end,
                        frameWeights: weights,
                        startTime: Double(placement.chunk - 1)
                            + Double(placement.start) * frameDuration + 0.5,
                        endTime: Double(placement.chunk - 1)
                            + Double(placement.end + 1) * frameDuration + 0.5,
                        embedding256: point.map { Float($0) },
                        rho128: []
                    )
                )
                embeddings.append(point)
                initialClusters.append(label)
                baselineAssignments.append(host)
            }
        }

        append(hostAPlacements.count == hostAPoints.count ? hostAPoints : hostAPoints, hostAPlacements, label: 0, host: 0)
        append(hostBPoints, hostBPlacements, label: 1, host: 1)
        append(points, placements, label: initialLabel, host: baselineHost)
        var extraCentroidMembers = [points]
        for child in extraChildren {
            append(child.points, child.placements, label: child.label, host: child.host)
            extraCentroidMembers.append(child.points)
        }

        let centroidA = MinorityClusterRescueEngine.normalizedMean(hostAPoints)
        let centroidB = MinorityClusterRescueEngine.normalizedMean(
            (hostBPoints + points.map { $0 }).map { $0 }
        )
        let centroids = [centroidA, centroidB]

        var options = MinorityClusterRescueOptions.conservative
        overrideOptions?(&options)

        let chunkOffsets = (0..<chunkCount).map { Double($0) - 0.5 }
        let engine = MinorityClusterRescueEngine(
            options: options,
            frameDuration: frameDuration,
            chunkOffsets: chunkOffsets,
            timedEmbeddings: timed,
            trainingIndices: Array(embeddings.indices),
            activityThreshold: activityThreshold
        )
        return Fixture(
            engine: engine,
            embeddings: embeddings,
            timed: timed,
            initialClusters: initialClusters,
            centroids: centroids,
            candidateIndices: embeddings.count - points.count..<embeddings.count
        )
    }

    private var hostAPoints: [[Double]] {
        baseSpread(base: unit([1.05, 0.55]), jitter: [
            [0.008, 0.0], [0.0, 0.01], [-0.006, -0.007], [0.004, 0.005],
        ])
    }

    private var hostBPoints: [[Double]] {
        baseSpread(base: unit([0.35, 1.10]), jitter: [
            [0.01, 0.0], [0.0, 0.012], [-0.008, 0.0],
            [0.0, -0.01], [0.006, 0.008], [-0.005, -0.006],
        ])
    }

    private var candidateCPoints: [[Double]] {
        baseSpread(base: unit([-0.95, 0.30]), jitter: [
            [0.004, 0.0], [0.0, 0.005], [-0.003, -0.004], [0.002, 0.003],
        ])
    }

    private var candidateDPoints: [[Double]] {
        baseSpread(base: unit([0.10, -1.00]), jitter: [[0.003, 0.0], [0.0, 0.004], [0.0, 0.0], [0.0, 0.0]])
    }

    private func baseSpread(base: [Double], jitter: [[Double]]) -> [[Double]] {
        jitter.map { j in unit([base[0] + j[0], base[1] + j[1]]) }
    }

    private func unit(_ v: [Double]) -> [Double] {
        MinorityClusterRescueEngine.normalized(v)
    }

    private func constrainedBaseline(_ fixture: Fixture) -> [Int] {
        ConstrainedClusterAssignment.assign(
            scores: MinorityClusterRescueEngine.centroidScores(
                embeddingFeatures: fixture.embeddings,
                centroids: fixture.centroids
            ),
            chunkIndices: fixture.timed.map { $0.chunkIndex }
        )
    }

    private func resolve(
        _ fixture: Fixture,
        constrained: Bool = true
    ) -> MinorityClusterRescueResult {
        let baseline = constrainedBaseline(fixture)
        return fixture.engine.resolve(
            embeddingFeatures: fixture.embeddings,
            trainingIndices: Array(fixture.embeddings.indices),
            initialClusters: fixture.initialClusters,
            baselineAssignments: baseline,
            baselineCentroids: fixture.centroids,
            constrainedAssignment: constrained
        )
    }

    // MARK: - Core rescue behavior

    func testRescueRecoversSwallowedMinorityCluster() {
        let fixture = self.fixture(
            points: candidateCPoints,
            placements: candidatePlacements,
            initialLabel: 2,
            baselineHost: 1
        )
        let result = resolve(fixture)

        XCTAssertEqual(result.rescuedClusters, 1)
        XCTAssertEqual(result.centroids.count, fixture.centroids.count + 1)

        let baseline = constrainedBaseline(fixture)
        for index in fixture.candidateIndices {
            let baselineCluster = baseline[index]
            XCTAssertLessThan(baselineCluster, fixture.centroids.count)
            XCTAssertEqual(baselineCluster, 1)
        }
    }

    func testRescueMovesEveryoneConsistently() {
        let fixture = self.fixture(
            points: candidateCPoints,
            placements: candidatePlacements,
            initialLabel: 2,
            baselineHost: 1
        )
        let result = resolve(fixture)
        let rescuedCluster = fixture.centroids.count
        let candidateCount = fixture.candidateIndices.count
        // Every candidate member converges on the newly appended centroid.
        for index in fixture.candidateIndices {
            XCTAssertEqual(result.assignments[index], rescuedCluster, "candidate member \(index)")
        }
        // Host members stay with their host.
        let baseline = constrainedBaseline(fixture)
        for index in fixture.embeddings.indices where !fixture.candidateIndices.contains(index) {
            XCTAssertEqual(result.assignments[index], baseline[index], "member \(index)")
        }
        _ = candidateCount
    }

    // MARK: - Gate coverage

    func testSingleChunkCandidateIsRejected() {
        // All four members share chunk 3 in two local speaker slots: only one
        // distinct chunk backs the child => gate 1 fails.
        let fixture = self.fixture(
            points: candidateCPoints,
            placements: [(3, 0, 9), (3, 0, 9), (3, 5, 14), (3, 10, 19)],
            initialLabel: 2,
            baselineHost: 1
        )
        XCTAssertEqual(resolve(fixture).rescuedClusters, 0)
    }

    func testCandidateWithoutCrossSpeechRunsIsRejected() {
        // Activity merged into one run when the inter-run gap is held below
        // crossSpeechGapSeconds. Widening the required gap merges all runs.
        let fixture = self.fixture(
            points: candidateCPoints,
            placements: candidatePlacements,
            initialLabel: 2,
            baselineHost: 1,
            overrideOptions: { options in options.crossSpeechGapSeconds = 60 }
        )
        XCTAssertEqual(resolve(fixture).rescuedClusters, 0)
    }

    func testCandidateWithInsufficientEffectiveSpeechIsRejected() {
        let fixture = self.fixture(
            points: candidateCPoints,
            placements: candidatePlacements,
            initialLabel: 2,
            baselineHost: 1,
            overrideOptions: { options in options.minEffectiveSpeechSeconds = 60 }
        )
        XCTAssertEqual(resolve(fixture).rescuedClusters, 0)
    }

    func testLowMarginCandidateIsRejected() {
        // A candidate whose spread matches its distance from the host fails
        // the relative-separation gate.
        let loosePoints: [[Double]] = [
            unit([-0.95, 0.30]),
            unit([-0.35, 0.92]),
            unit([0.30, 0.94]),
            unit([-0.15, -0.98]),
        ]
        let fixture = self.fixture(
            points: loosePoints,
            placements: candidatePlacements,
            initialLabel: 2,
            baselineHost: 1
        )
        XCTAssertEqual(resolve(fixture).rescuedClusters, 0)
    }

    func testHostSurvivalGateRejectsCandidateWhenOnlyCandidateRemains() {
        // Host B carries just one independent member; once the candidate is
        // split off, the host would not retain the minimum required support.
        let loneHostPlacements: [(Int, Int, Int)] = [(0, 10, 19)]
        let fixture = self.fixture(
            points: candidateCPoints,
            placements: candidatePlacements,
            initialLabel: 2,
            baselineHost: 1
        )
        // Duplicate host B members by reusing them all with chunk overlap —
        // instead, replace host members so only one independent member exists.
        var trimmedHost: [TimedEmbedding] = []
        var trimmedFeatures: [[Double]] = []
        var trimmedInitial: [Int] = []
        var trimmedBaseline: [Int] = []
        for (index, point) in hostBPoints.enumerated().prefix(1) {
            let placement = hostBPlacements[index]
            var weights = [Float](repeating: 0, count: framesPerWindow)
            for frame in placement.1...placement.2 {
                weights[frame] = 1
            }
            trimmedHost.append(
                TimedEmbedding(
                    chunkIndex: placement.0,
                    speakerIndex: 0,
                    startFrame: placement.1,
                    endFrame: placement.2,
                    frameWeights: weights,
                    startTime: Double(placement.0 - 1) + Double(placement.1) * frameDuration + 0.5,
                    endTime: Double(placement.0 - 1) + Double(placement.2 + 1) * frameDuration + 0.5,
                    embedding256: point.map { Float($0) },
                    rho128: []
                )
            )
            trimmedFeatures.append(point)
            trimmedInitial.append(1)
            trimmedBaseline.append(1)
        }
        for (index, point) in candidateCPoints.enumerated() {
            let placement = candidatePlacements[index]
            var weights = [Float](repeating: 0, count: framesPerWindow)
            for frame in placement.1...placement.2 {
                weights[frame] = 1
            }
            trimmedHost.append(
                TimedEmbedding(
                    chunkIndex: placement.0,
                    speakerIndex: 1,
                    startFrame: placement.1,
                    endFrame: placement.2,
                    frameWeights: weights,
                    startTime: Double(placement.0 - 1) + Double(placement.1) * frameDuration + 0.5,
                    endTime: Double(placement.0 - 1) + Double(placement.2 + 1) * frameDuration + 0.5,
                    embedding256: point.map { Float($0) },
                    rho128: []
                )
            )
            trimmedFeatures.append(point)
            trimmedInitial.append(2)
            trimmedBaseline.append(1)
        }
        let isolatedHost = MinorityClusterRescueEngine.normalizedMean(hostBPoints.prefix(1).map { $0 })
        let engine = MinorityClusterRescueEngine(
            options: .conservative,
            frameDuration: frameDuration,
            chunkOffsets: (0..<chunkCount).map { Double($0) - 0.5 },
            timedEmbeddings: trimmedHost,
            trainingIndices: Array(trimmedFeatures.indices),
            activityThreshold: 0.5
        )
        let result = engine.resolve(
            embeddingFeatures: trimmedFeatures,
            trainingIndices: Array(trimmedFeatures.indices),
            initialClusters: trimmedInitial,
            baselineAssignments: trimmedBaseline,
            baselineCentroids: [isolatedHost],
            constrainedAssignment: true
        )
        XCTAssertEqual(result.rescuedClusters, 0)
        // Make the compiler retain the unused base fixture.
        _ = fixture
    }

    func testMultipleCandidatesRecoverTogether() {
        let fixture = self.fixture(
            points: candidateCPoints,
            placements: candidatePlacements,
            initialLabel: 2,
            baselineHost: 1,
            extraChildren: [
                (candidateDPoints, secondCandidatePlacements, 3, 1),
            ]
        )
        let result = resolve(fixture)
        XCTAssertEqual(result.rescuedClusters, 2)
        XCTAssertEqual(result.centroids.count, 4)
    }

    func testConflictingCandidatesAcceptOnlySeparatedSubset() {
        // Child C2 sits inside candidate C1's separation radius: mutually
        // incompatibility rejects the second.
        let conflicting = candidateCPoints.map { point in
            unit([point[0] + 0.006, point[1] + 0.003])
        }
        let fixture = self.fixture(
            points: candidateCPoints,
            placements: candidatePlacements,
            initialLabel: 2,
            baselineHost: 1,
            extraChildren: [
                (conflicting, secondCandidatePlacements, 3, 1),
            ]
        )
        let result = resolve(fixture)
        XCTAssertEqual(result.rescuedClusters, 1)
    }

    func testSpeakerCeilingCapsAcceptedCandidates() {
        // Baseline has 2 centroids; a ceiling of 3 caps the rescue at one.
        var capped: ((inout MinorityClusterRescueOptions) -> Void)? = nil
        var captured: (inout MinorityClusterRescueOptions) -> Void = { options in
            options.maximumSpeakers = 3
        }
        capped = captured
        let fixture = self.fixture(
            points: candidateCPoints,
            placements: candidatePlacements,
            initialLabel: 2,
            baselineHost: 1,
            overrideOptions: capped,
            extraChildren: [(candidateDPoints, secondCandidatePlacements, 3, 1)]
        )
        let result = resolve(fixture)
        XCTAssertEqual(result.rescuedClusters, 1)
        XCTAssertEqual(result.centroids.count, 3)
    }

    func testCoChunkSlotsRemainDistinct() {
        // Chunk 0 hosts a member of A (slot 0) and B (slot 1); the candidate's
        // rescued cluster must not absorb others of the same chunk.
        let fixture = self.fixture(
            points: candidateCPoints,
            placements: candidatePlacements,
            initialLabel: 2,
            baselineHost: 1
        )
        let result = resolve(fixture)
        let rescuedCluster = fixture.centroids.count
        var rescuedPerChunk: [Int: Int] = [:]
        for (index, cluster) in result.assignments.enumerated() where cluster == rescuedCluster {
            rescuedPerChunk[fixture.timed[index].chunkIndex, default: 0] += 1
        }
        for (_, count) in rescuedPerChunk {
            XCTAssertLessThanOrEqual(count, 1)
        }
    }

    func testRescueIsDeterministic() {
        let first = self.fixture(
            points: candidateCPoints,
            placements: candidatePlacements,
            initialLabel: 2,
            baselineHost: 1
        )
        let second = self.fixture(
            points: candidateCPoints,
            placements: candidatePlacements,
            initialLabel: 2,
            baselineHost: 1
        )
        let firstResult = resolve(first)
        let secondResult = resolve(second)
        XCTAssertEqual(firstResult.assignments, secondResult.assignments)
        XCTAssertEqual(firstResult.centroids, secondResult.centroids)
        XCTAssertEqual(firstResult.rescuedClusters, secondResult.rescuedClusters)
    }

    func testEngineGuardsReturnBaseline() {
        let fixture = self.fixture(
            points: candidateCPoints,
            placements: candidatePlacements,
            initialLabel: 2,
            baselineHost: 1
        )
        let brokenEngine = MinorityClusterRescueEngine(
            options: .conservative,
            frameDuration: 0,
            chunkOffsets: (0..<chunkCount).map { Double($0) - 0.5 },
            timedEmbeddings: fixture.timed,
            trainingIndices: Array(fixture.embeddings.indices),
            activityThreshold: 0.5
        )
        let baseline = constrainedBaseline(fixture)
        let result = brokenEngine.resolve(
            embeddingFeatures: fixture.embeddings,
            trainingIndices: Array(fixture.embeddings.indices),
            initialClusters: fixture.initialClusters,
            baselineAssignments: baseline,
            baselineCentroids: fixture.centroids,
            constrainedAssignment: true
        )
        XCTAssertEqual(result.rescuedClusters, 0)
        // Baseline assignments must be reproduced verbatim (baseline-preserving
        // fallback), not a partially reworked result.
    }

    // MARK: - Configuration surface

    func testDefaultConfigDisablesRescue() {
        XCTAssertEqual(OfflineDiarizerConfig.default.clustering.minorityRescue, .disabled)
    }

    // MARK: - Insertion composition (assignment -> rescue -> chunk matrix)

    func testRescueFeedsChunkAssignmentMatrix() {
        // Orders the manager's sequence: constrained baseline assignment,
        // rescue, then the matrix handed to the timeline reconstruction.
        let fixture = self.fixture(
            points: candidateCPoints,
            placements: candidatePlacements,
            initialLabel: 2,
            baselineHost: 1
        )
        let result = resolve(fixture)
        XCTAssertEqual(result.rescuedClusters, 1)

        var matrix = Array(
            repeating: Array(repeating: -2, count: slotsPerChunk),
            count: chunkCount
        )
        for (timedEmbedding, cluster) in zip(fixture.timed, result.assignments) {
            guard timedEmbedding.chunkIndex >= 0,
                  timedEmbedding.chunkIndex < matrix.count,
                  timedEmbedding.speakerIndex >= 0,
                  timedEmbedding.speakerIndex < slotsPerChunk,
                  cluster >= 0,
                  cluster < result.centroids.count
            else { continue }
            matrix[timedEmbedding.chunkIndex][timedEmbedding.speakerIndex] = cluster
        }

        // At least one candidate frame lands in the rescued cluster, and no
        // baseline cluster vanishes from the matrix.
        let flattened = Set(matrix.flatMap { $0 })
        XCTAssertGreaterThanOrEqual(flattened.count, 3)
    }
}

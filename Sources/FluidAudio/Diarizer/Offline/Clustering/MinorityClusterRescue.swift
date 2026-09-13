import Foundation

/// Opt-in rescue of AHC minority child clusters that FluidAudio's VB/PLDA
/// refinement stage swallowed into their majority host cluster.
///
/// Off by default (`MinorityClusterRescue.disabled`): callers that never set
/// `config.clustering.minorityRescue` observe upstream-identical behavior.
///
/// The rescue never re-clusters. It compares the AHC `initialClusters` produced
/// from the raw 256d embeddings against the baseline assignments and, when an
/// AHC child has strong independent, cross-turn, geometrically separated support
/// that the baseline absorbed into one host, appends the child's centroid and
/// re-runs the existing (constrained) assignment pass. Every gate in
/// `MinorityClusterRescueOptions` must hold, otherwise the baseline result is
/// returned unchanged (baseline-preserving fallback).
@available(macOS 14.0, iOS 17.0, *)
public enum MinorityClusterRescue: Sendable, Equatable {
    /// Upstream-compatible default: never run the rescue.
    case disabled
    /// Run the rescue with the given conservative gates.
    case enabled(MinorityClusterRescueOptions)

    /// The conservative production gate profile.
    public static let conservative = MinorityClusterRescue.enabled(
        .conservative
    )
}

/// Thresholds for the conservative minority-cluster rescue. Every gate must
/// pass for a candidate to be rescued; tuning changes should be benchmarked.
@available(macOS 14.0, iOS 17.0, *)
public struct MinorityClusterRescueOptions: Sendable, Equatable {
    /// `nearestHostDistance / max(candidateWithinSpread, epsilon)` must be at
    /// least this ratio (dimensionless Euclidean margin on unit-normalized
    /// embeddings).
    public var rescueMargin: Double
    /// Minimum temporal gap (seconds) between distinct sustained activity
    /// intervals supporting one candidate.
    public var crossSpeechGapSeconds: Double
    /// Minimum canonical, deduplicated speech support (seconds).
    public var minEffectiveSpeechSeconds: Double
    /// Minimum number of distinct, non-overlapping chunks backing a candidate.
    public var minIndependentChunks: Int
    /// Hard product ceiling on the number of speakers after any rescue.
    public var maximumSpeakers: Int
    /// Fraction of a candidate cluster's effectively-supported members that a
    /// single baseline host must have absorbed for the rescue to be eligible.
    public var hostAbsorptionFraction: Double

    public init(
        rescueMargin: Double,
        crossSpeechGapSeconds: Double,
        minEffectiveSpeechSeconds: Double,
        minIndependentChunks: Int,
        maximumSpeakers: Int,
        hostAbsorptionFraction: Double
    ) {
        self.rescueMargin = rescueMargin
        self.crossSpeechGapSeconds = crossSpeechGapSeconds
        self.minEffectiveSpeechSeconds = minEffectiveSpeechSeconds
        self.minIndependentChunks = minIndependentChunks
        self.maximumSpeakers = maximumSpeakers
        self.hostAbsorptionFraction = hostAbsorptionFraction
    }

    public static let conservative = MinorityClusterRescueOptions(
        rescueMargin: 1.5,
        crossSpeechGapSeconds: 2.0,
        minEffectiveSpeechSeconds: 2.0,
        minIndependentChunks: 2,
        maximumSpeakers: 8,
        hostAbsorptionFraction: 0.9
    )
}

/// Result of an (attempted) rescue pass: either the inherited baseline or a
/// result with appended centroids and recomputed assignments.
@available(macOS 14.0, iOS 17.0, *)
struct MinorityClusterRescueResult {
    /// Cluster index per embedding, parallel to the full embedding space.
    var assignments: [Int]
    /// Centroids including any appended rescued clusters.
    var centroids: [[Double]]
    /// Number of successfully rescued candidates (0 = baseline passthrough).
    var rescuedClusters: Int
}

@available(macOS 14.0, iOS 17.0, *)
struct MinorityClusterRescueEngine {
    private let logger = AppLogger(category: "OfflineMinorityRescue")

    struct Candidate {
        /// AHC label of the child cluster.
        var initialClusterLabel: Int
        /// Baseline host (centroid index) that absorbed the child.
        var hostIndex: Int
        /// Training-space member indices of the child.
        var members: [Int]
        /// Unit-normalized centroid of the candidate members.
        var centroid: [Double]
        var withinSpread: Double
        var distinctChunks: Int
        var effectiveSpeechSeconds: Double
    }

    let options: MinorityClusterRescueOptions
    /// Canonical frame duration (seconds); 0 disables the rescue.
    let frameDuration: Double
    /// Per-chunk start time in seconds; the quantization anchor shared with the
    /// offline reconstruction (`OfflineReconstruction` uses the same rule).
    let chunkOffsets: [Double]
    /// Full `TimedEmbedding` list (indexed by the full embedding space).
    let timedEmbeddings: [TimedEmbedding]
    /// Embedding weights above this count as speech activity, matching the
    /// reconstruction's "activation > 0 counts a frame" semantics.
    let activityThreshold: Float
    /// Training index -> full embedding index.
    let trainingIndices: [Int]

    init(
        options: MinorityClusterRescueOptions,
        frameDuration: Double,
        chunkOffsets: [Double],
        timedEmbeddings: [TimedEmbedding],
        trainingIndices: [Int],
        activityThreshold: Float = 0
    ) {
        self.options = options
        self.frameDuration = frameDuration
        self.chunkOffsets = chunkOffsets
        self.timedEmbeddings = timedEmbeddings
        self.trainingIndices = trainingIndices
        self.activityThreshold = activityThreshold
    }

    // MARK: - Entry point

    /// Technical-only diagnostics: labels, counts and margin ratios. Never
    /// audio, text, paths, embedding contents, or stable identifiers.
    private func log(_ level: AppLogLevel, _ text: String) {
        switch level {
        case .debug: logger.debug(text)
        case .info: logger.info(text)
        }
    }

    enum AppLogLevel { case debug, info }

    /// Applies the rescue gate cascade to the baseline clustering result.
    ///
    /// `initialClusters` and the returned indices describe the training subset
    /// via `trainingIndices` (training index -> full embedding index);
    /// `baselineAssignments` covers the full embedding space. Any violated
    /// gate or invariant returns the baseline unchanged.
    func resolve(
        embeddingFeatures: [[Double]],
        trainingIndices: [Int],
        initialClusters: [Int],
        baselineAssignments: [Int],
        baselineCentroids: [[Double]],
        constrainedAssignment: Bool
    ) -> MinorityClusterRescueResult {
        let baseline = MinorityClusterRescueResult(
            assignments: baselineAssignments,
            centroids: baselineCentroids,
            rescuedClusters: 0
        )

        guard frameDuration > 0,
              !baselineCentroids.isEmpty,
              !initialClusters.isEmpty,
              trainingIndices.count == initialClusters.count,
              trainingIndices.count <= timedEmbeddings.count,
              embeddingFeatures.count == timedEmbeddings.count,
              embeddingFeatures.count == baselineAssignments.count
        else {
            return baseline
        }

        let candidates = buildCandidates(
            embeddingFeatures: embeddingFeatures,
            trainingIndices: trainingIndices,
            initialClusters: initialClusters,
            baselineAssignments: baselineAssignments,
            baselineCentroids: baselineCentroids
        )
        guard !candidates.isEmpty else { return baseline }

        let accepted = selectCompatibleCandidates(
            candidates,
            baselineCentroidCount: baselineCentroids.count,
            baselineAssignments: baselineAssignments,
            trainingIndices: trainingIndices,
            embeddingFeatures: embeddingFeatures
        )
        guard !accepted.isEmpty else { return baseline }

        var centroids = baselineCentroids
        // Deterministic append order: candidates sorted by (host, label, centroid).
        for candidate in accepted where centroids.count < options.maximumSpeakers {
            centroids.append(candidate.centroid)
        }
        guard centroids.count > baselineCentroids.count else { return baseline }

        let scores = Self.centroidScores(
            embeddingFeatures: embeddingFeatures,
            centroids: centroids
        )
        let assignments: [Int]
        if constrainedAssignment {
            assignments = ConstrainedClusterAssignment.assign(
                scores: scores,
                chunkIndices: timedEmbeddings.map(\.chunkIndex)
            )
        } else {
            assignments = scores.map { row in
                var bestIndex = 0
                var bestScore = -Double.infinity
                for (index, score) in row.enumerated() where score > bestScore {
                    bestScore = score
                    bestIndex = index
                }
                return bestIndex
            }
        }

        // Invariant: the reassignment still covers the full embedding space.
        guard assignments.count == embeddingFeatures.count,
              assignments.contains(where: { $0 >= 0 })
        else {
            return baseline
        }

        return MinorityClusterRescueResult(
            assignments: assignments,
            centroids: centroids,
            rescuedClusters: centroids.count - baselineCentroids.count
        )
    }

    // MARK: - Candidate generation

    private func buildCandidates(
        embeddingFeatures: [[Double]],
        trainingIndices: [Int],
        initialClusters: [Int],
        baselineAssignments: [Int],
        baselineCentroids: [[Double]]
    ) -> [Candidate] {
        var membersByCluster: [Int: [Int]] = [:]
        for (trainingIndex, label) in initialClusters.enumerated() {
            membersByCluster[label, default: []].append(trainingIndex)
        }

        var candidates: [Candidate] = []
        for label in membersByCluster.keys.sorted() {
            if let candidate = evaluate(
                label: label,
                members: membersByCluster[label] ?? [],
                embeddingFeatures: embeddingFeatures,
                trainingIndices: trainingIndices,
                baselineAssignments: baselineAssignments,
                baselineCentroids: baselineCentroids
            ) {
                candidates.append(candidate)
            }
        }
        return candidates
    }

    private func trainingBaseline(
        _ trainingIndex: Int,
        trainingIndices: [Int],
        baselineAssignments: [Int]
    ) -> Int {
        let embeddingIndex = trainingIndices[trainingIndex]
        guard embeddingIndex < baselineAssignments.count else { return -1 }
        return baselineAssignments[embeddingIndex]
    }

    private func evaluate(
        label: Int,
        members: [Int],
        embeddingFeatures: [[Double]],
        trainingIndices: [Int],
        baselineAssignments: [Int],
        baselineCentroids: [[Double]]
    ) -> Candidate? {
        guard members.count >= 2 else { return nil }

        // Gate 1 — independent support: members backed by at least
        // `minIndependentChunks` distinct chunk indices whose source windows do
        // not overlap in time; same-chunk speaker slots are not independent
        // evidence.
        var perChunk: [Int: [(start: Double, end: Double)]] = [:]
        let activeMembers = members.filter { trainingIndex in
            candidateSupport(trainingIndex: trainingIndex).count > 0
        }
        guard !activeMembers.isEmpty else 
            
            { log(.debug, "gate1 no active members"); return nil }
        for trainingIndex in activeMembers {
            let embedding = timedEmbeddings[trainingIndices[trainingIndex]]
            let span = windowSpan(trainingIndex: trainingIndex)
            perChunk[embedding.chunkIndex, default: []].append(span)
        }
        let distinctChunks = perChunk.keys.filter { index in index >= 0 }
        guard distinctChunks.count >= options.minIndependentChunks else 
            
            { log(.debug, "gate1 insufficient distinct chunks"); return nil }
        // Intervals across distinct chunks must not overlap.
        let acrossChunkIntervals = perChunk.keys
            .filter { $0 >= 0 }
            .compactMap { chunkIndex -> (start: Double, end: Double)? in
                guard chunkIndex < chunkOffsets.count else { return nil }
                let spans = perChunk[chunkIndex] ?? []
                guard let first = spans.map(\.start).min(),
                      let last = spans.map(\.end).max()
                else { return nil }
                return (first, last)
            }
        guard
            intervalsAreMutuallyDisjoint(acrossChunkIntervals)
        else { return nil }

        // Gate 1b — absorption shape: a strong fraction of the candidate must
        // have been absorbed by one baseline host.
        var hostsByMember: [Int: [Int]] = [:]
        var activeCarrier = 0
        for trainingIndex in activeMembers {
            let host = trainingBaseline(
                trainingIndex,
                trainingIndices: trainingIndices,
                baselineAssignments: baselineAssignments
            )
            guard host >= 0 else { continue }
            hostsByMember[host, default: []].append(trainingIndex)
            activeCarrier += 1
        }
        guard activeCarrier > 0, let dominantHost = hostsByMember.max(by: {
            if $0.value.count == $1.value.count {
                // Deterministic tie-break on host label.
                return $0.key < $1.key
            }
            return $0.value.count < $1.value.count
        }),
        Double(dominantHost.value.count) / Double(activeCarrier) >= options.hostAbsorptionFraction
        else { return nil }
        let host = dominantHost.key

        // Gate 2 — canonical frames (deduplicated across overlapping windows)
        // must form at least two sustained runs separated by
        // `crossSpeechGapSeconds`.
        var candidateFrames: Set<Int> = []
        for trainingIndex in members {
            candidateFrames.formUnion(candidateSupport(trainingIndex: trainingIndex))
        }
        let activityRuns = Self.frameRuns(
            frames: candidateFrames.sorted(),
            frameDuration: frameDuration,
            gapThreshold: options.crossSpeechGapSeconds
        )
        guard activityRuns.count >= 2 else 
            
            { log(.debug, "gate2 insufficient cross-speech runs"); return nil }
        let effectiveSpeechSeconds = Double(candidateFrames.count) * frameDuration
        guard effectiveSpeechSeconds >= options.minEffectiveSpeechSeconds else 
            
            { log(.debug, "gate3 insufficient effective speech"); return nil }

        // Gate 3 — relative separation: the candidate centroid must be far
        // enough from the host's remaining embeddings relative to its own
        // spread (nearestHostDistance / spread >= rescueMargin).
        let features = members.map { embeddingFeatures[trainingIndices[$0]] }
        guard let dimension = features.first?.count, dimension > 0, features.allSatisfy({ $0.count == dimension }) else {
            return nil
        }
        let centroid = Self.normalizedMean(features)
        guard !centroid.isEmpty else { return nil }
        var withinSpread = 0.0
        for feature in features {
            withinSpread = max(withinSpread, Self.distance(feature, centroid))
        }
        guard withinSpread.isFinite else { return nil }

        // Gate 3a — the child must be distinct from the host identity itself;
        // rescuing the host's own initial cluster would duplicate an existing
        // speaker instead of recovering a swallowed child.
        guard host >= 0, host < baselineCentroids.count, !baselineCentroids[host].isEmpty else {
            return nil
        }
        let hostCentroid = Self.normalized(baselineCentroids[host])
        let identityRatio = Self.distance(centroid, hostCentroid) / max(withinSpread, 1e-6)
        if identityRatio < options.rescueMargin {
            log(.debug, "gate3a insufficient identity separation ratio=\(identityRatio)")
            return nil
        }

        return Candidate(
            initialClusterLabel: label,
            hostIndex: host,
            members: members,
            centroid: centroid,
            withinSpread: withinSpread,
            distinctChunks: distinctChunks.count,
            effectiveSpeechSeconds: effectiveSpeechSeconds
        )
    }

    /// Time span of a training member's embedding window footprint.
    private func windowSpan(trainingIndex: Int) -> (start: Double, end: Double) {
        let embeddingIndex = embeddingIndexOf(trainingIndex)
        guard embeddingIndex >= 0, embeddingIndex < timedEmbeddings.count else {
            return (0, 0)
        }
        let embedding = timedEmbeddings[embeddingIndex]
        guard embedding.chunkIndex >= 0, embedding.chunkIndex < chunkOffsets.count else {
            return (0, 0)
        }
        let offset = chunkOffsets[embedding.chunkIndex]
        let start = offset + Double(embedding.startFrame) * frameDuration
        let end = offset + Double(embedding.endFrame + 1) * frameDuration
        return (start, end)
    }
    // MARK: - Compatible candidate selection

    private func selectCompatibleCandidates(
        _ candidates: [Candidate],
        baselineCentroidCount: Int,
        baselineAssignments: [Int],
        trainingIndices: [Int],
        embeddingFeatures: [[Double]]
    ) -> [Candidate] {
        var accepted: [Candidate] = []

        let hostGrouping = Dictionary(grouping: candidates) { $0.hostIndex }
        for host in hostGrouping.keys.sorted() {
            let hostCandidates = (hostGrouping[host] ?? []).sorted { lhs, rhs in
                if lhs.initialClusterLabel != rhs.initialClusterLabel {
                    return lhs.initialClusterLabel < rhs.initialClusterLabel
                }
                return lhs.centroid.lexicographicallyPrecedes(rhs.centroid)
            }

            // Joint exclusion across every qualifying candidate of this host:
            // the "remaining host" is what stays when all accepted sibling
            // children are split off. This keeps the host-remaining margin
            // fair for each candidate regardless of selection order.
            let jointlyRemoved = Set(hostCandidates.flatMap { $0.members })

            var keptForHost: [Candidate] = []
            for candidate in hostCandidates {
                // Gate 6 — product ceiling on total speakers: stop accepting
                // candidates before appending beyond `maximumSpeakers`.
                guard baselineCentroidCount + accepted.count < options.maximumSpeakers else {
                    break
                }

                // Gate 3b — stable separation from the host's jointly
                // remaining members.
                var nearestRemaining = Double.infinity
                for trainingIndex in trainingIndices.indices where !jointlyRemoved.contains(trainingIndex) {
                    let embeddingIndex = trainingIndices[trainingIndex]
                    guard embeddingIndex < baselineAssignments.count,
                          baselineAssignments[embeddingIndex] == host
                    else { continue }
                    let delta = Self.distance(
                        embeddingFeatures[embeddingIndex],
                        candidate.centroid
                    )
                    nearestRemaining = min(nearestRemaining, delta)
                }
                let remainingRatio = nearestRemaining / max(candidate.withinSpread, 1e-6)
                if !nearestRemaining.isFinite || remainingRatio < options.rescueMargin {
                    logger.debug("gate3b insufficient remaining-host margin ratio=\(remainingRatio)")
                    continue
                }

                // Gate 4 — host remaining support after splitting off the full
                // qualifying candidate set (prevents the host from being
                // silently renamed into a new speaker).
                guard
                    hostSurvivesAfterRemoval(
                        host: host,
                        removed: Array(jointlyRemoved),
                        baselineAssignments: baselineAssignments,
                        trainingIndices: trainingIndices
                    )
                else { continue }

                // Gate 5 — candidate centroids must be mutually separated (a
                // pair of rescued centroids cannot be the same speaker).
                var separatedFromPeers = true
                for other in keptForHost {
                    let spread = max(candidate.withinSpread, other.withinSpread)
                    let ratio = Self.distance(candidate.centroid, other.centroid)
                        / max(spread, 1e-6)
                    if ratio < options.rescueMargin {
                        separatedFromPeers = false
                        break
                    }
                }
                if !separatedFromPeers {
                    logger.debug("gate5 candidate peer conflict")
                    continue
                }

                keptForHost.append(candidate)
            }
            accepted.append(contentsOf: keptForHost)
        }

        return accepted
    }

    private func hostSurvivesAfterRemoval(
        host: Int,
        removed: [Int],
        baselineAssignments: [Int],
        trainingIndices: [Int]
    ) -> Bool {
        let removedSet = Set(removed)
        var survivingChunks: [Int] = []
        var survivingFrames: Set<Int> = []
        var survivingCount = 0
        for trainingIndex in trainingIndices.indices where !removedSet.contains(trainingIndex) {
            let embeddingIndex = trainingIndices[trainingIndex]
            guard embeddingIndex < baselineAssignments.count,
                  baselineAssignments[embeddingIndex] == host
            else { continue }
            survivingCount += 1
            survivingChunks.append(timedEmbeddings[embeddingIndex].chunkIndex)
            survivingFrames.formUnion(support(trainingIndex: trainingIndex))
        }
        guard survivingCount >= options.minIndependentChunks,
              Set(survivingChunks).count >= options.minIndependentChunks,
              Double(survivingFrames.count) * frameDuration >= options.minEffectiveSpeechSeconds
        else {
            log(.debug, "gate4 host remaining support failed")
            return false
        }
        return true
    }

    // MARK: - Per-embedding timeline helpers

    /// Canonical speech frames of a training member (seconds quantization).
    private func support(trainingIndex: Int) -> Set<Int> {
        candidateSupport(trainingIndex: trainingIndex)
    }

    private func candidateSupport(trainingIndex: Int) -> Set<Int> {
        let embeddingIndex = embeddingIndexOf(trainingIndex)
        guard embeddingIndex >= 0, embeddingIndex < timedEmbeddings.count else { return [] }
        let embedding = timedEmbeddings[embeddingIndex]
        guard embedding.chunkIndex < chunkOffsets.count else { return [] }
        let chunkOffset = chunkOffsets[embedding.chunkIndex]
        var frames: Set<Int> = []
        for (localFrame, weight) in embedding.frameWeights.enumerated()
        where weight > activityThreshold {
            let globalFrame = max(
                0,
                Int(((chunkOffset + Double(localFrame) * frameDuration) / frameDuration).rounded())
            )
            frames.insert(globalFrame)
        }
        return frames
    }

    /// Groups canonical frames into sustained runs; a new run starts whenever
    /// the gap between consecutive frames reaches `gapThreshold` (seconds).
    static func frameRuns(
        frames: [Int],
        frameDuration: Double,
        gapThreshold: Double
    ) -> [(start: Double, end: Double)] {
        guard frameDuration > 0, !frames.isEmpty else { return [] }
        var runs: [(start: Double, end: Double)] = []
        var lastFrame: Int?
        for frame in frames {
            let start = Double(frame) * frameDuration
            let end = start + frameDuration
            if let lastFrame,
                start - (Double(lastFrame + 1) * frameDuration) < gapThreshold
            {
                runs[runs.count - 1].end = max(runs[runs.count - 1].end, end)
            } else {
                runs.append((start, end))
            }
            lastFrame = frame
        }
        return runs
    }


    private func intervalsAreMutuallyDisjoint(
        _ intervals: [(start: Double, end: Double)]
    ) -> Bool {
        let sorted = intervals.sorted { $0.start < $1.start }
        for index in 1..<sorted.count where sorted[index].start < sorted[index - 1].end {
            return false
        }
        return true
    }

    private func embeddingIndexOf(_ trainingIndex: Int) -> Int {
        trainingIndices.indices.contains(trainingIndex)
            ? trainingIndices[trainingIndex]
            : -1
    }

    // MARK: - Geometry helpers

    static func centroidScores(
        embeddingFeatures: [[Double]],
        centroids: [[Double]]
    ) -> [[Double]] {
        let normalizedCentroids = centroids.map(normalized)
        return embeddingFeatures.map { feature in
            let normalizedFeature = normalized(feature)
            return normalizedCentroids.map { dot(normalizedFeature, $0) }
        }
    }

    static func normalized(_ vector: [Double]) -> [Double] {
        guard !vector.isEmpty else { return vector }
        var sumSquares = 0.0
        for value in vector where value.isFinite {
            sumSquares += value * value
        }
        guard sumSquares.isFinite, sumSquares > 0 else { return vector }
        let scale = 1 / sqrt(sumSquares)
        return vector.map { $0 * scale }
    }

    static func dot(_ lhs: [Double], _ rhs: [Double]) -> Double {
        guard lhs.count == rhs.count, !lhs.isEmpty else { return 0 }
        var result = 0.0
        for (left, right) in zip(lhs, rhs) {
            result += left * right
        }
        return result
    }

    static func normalizedMean(_ vectors: [[Double]]) -> [Double] {
        guard let dimension = vectors.first?.count, !vectors.isEmpty,
              vectors.allSatisfy({ $0.count == dimension })
        else { return [] }
        var mean = [Double](repeating: 0, count: dimension)
        for vector in vectors {
            for index in 0..<dimension {
                mean[index] += vector[index]
            }
        }
        let scale = 1 / Double(vectors.count)
        return normalized(mean.map { $0 * scale })
    }

    static func distance(_ lhs: [Double], _ rhs: [Double]) -> Double {
        guard lhs.count == rhs.count else { return .infinity }
        var squared = 0.0
        for (left, right) in zip(lhs, rhs) {
            let delta = left - right
            squared += delta * delta
        }
        return sqrt(max(0, squared))
    }
}

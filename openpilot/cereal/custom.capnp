using Cxx = import "/include/c++.capnp";
$Cxx.namespace("cereal");

@0xb526ba661d550a59;

# custom.capnp: a home for empty structs reserved for custom forks
# These structs are guaranteed to remain reserved and empty in mainline
# cereal, so use these if you want custom events in your fork.

# DO rename the structs
# DON'T change the identifier (e.g. @0x81c2f05a394cf4af)

struct ModularAssistiveDrivingSystem {
  state @0 :ModularAssistiveDrivingSystemState;
  enabled @1 :Bool;
  active @2 :Bool;
  available @3 :Bool;

  enum ModularAssistiveDrivingSystemState {
    disabled @0;
    paused @1;
    enabled @2;
    softDisabling @3;
    overriding @4;
  }
}

struct IntelligentCruiseButtonManagement {
  state @0 :IntelligentCruiseButtonManagementState;
  sendButton @1 :SendButtonState;
  vTarget @2 :Float32;

  enum IntelligentCruiseButtonManagementState {
    inactive @0;      # No button press or default state
    preActive @1;     # Pre-active state before transitioning to increasing or decreasing
    increasing @2;    # Increasing speed
    decreasing @3;    # Decreasing speed
    holding @4;       # Holding steady speed
  }

  enum SendButtonState {
    none @0;
    increase @1;
    decrease @2;
  }
}

# Same struct as Log.RadarState.LeadData
struct LeadData {
  dRel @0 :Float32;
  yRel @1 :Float32;
  vRel @2 :Float32;
  aRel @3 :Float32;
  vLead @4 :Float32;
  dPath @6 :Float32;
  vLat @7 :Float32;
  vLeadK @8 :Float32;
  aLeadK @9 :Float32;
  fcw @10 :Bool;
  status @11 :Bool;
  aLeadTau @12 :Float32;
  modelProb @13 :Float32;
  radar @14 :Bool;
  radarTrackId @15 :Int32 = -1;

  aLeadDEPRECATED @5 :Float32;
}

struct SelfdriveStateSP @0x81c2f05a394cf4af {
  mads @0 :ModularAssistiveDrivingSystem;
  intelligentCruiseButtonManagement @1 :IntelligentCruiseButtonManagement;
  buttonsPressed @2 :UInt16;
  buttonsReleaseToggle @3 :UInt16;

  enum AudibleAlert {
    none @0;

    engage @1;
    disengage @2;
    refuse @3;

    warningSoft @4;
    warningImmediate @5;

    prompt @6;
    promptRepeat @7;
    promptDistracted @8;

    # unused, these are reserved for upstream events so we don't collide
    reserved9 @9;
    reserved10 @10;
    reserved11 @11;
    reserved12 @12;
    reserved13 @13;
    reserved14 @14;
    reserved15 @15;
    reserved16 @16;
    reserved17 @17;
    reserved18 @18;
    reserved19 @19;
    reserved20 @20;
    reserved21 @21;
    reserved22 @22;
    reserved23 @23;
    reserved24 @24;
    reserved25 @25;
    reserved26 @26;
    reserved27 @27;
    reserved28 @28;
    reserved29 @29;
    reserved30 @30;

    promptSingleLow @31;
    promptSingleHigh @32;
  }
}

struct ModelManagerSP @0xaedffd8f31e7b55d {
  activeBundle @0 :ModelBundle;
  selectedBundle @1 :ModelBundle;
  availableBundles @2 :List(ModelBundle);

  struct DownloadUri {
    uri @0 :Text;
    sha256 @1 :Text;
  }

  enum DownloadStatus {
    notDownloading @0;
    downloading @1;
    downloaded @2;
    cached @3;
    failed @4;
    verifying @5;
  }

  struct DownloadProgress {
    status @0 :DownloadStatus;
    progress @1 :Float32;
    eta @2 :UInt32;
    speed @3 :Float32;
    canceling @4 :Bool;
    resuming @5 :Bool;
  }

  struct Chunk {
    fileName @0 :Text;
    sha256 @1 :Text;
  }

  struct Artifact {
    fileName @0 :Text;
    downloadUri @1 :DownloadUri;
    downloadProgress @2 :DownloadProgress;
    chunks @3 :List(Chunk);
  }

  struct Model {
    type @0 :Type;
    artifact @1 :Artifact;  # Main artifact
    metadata @2 :Artifact;  # Metadata artifact

    enum Type {
      supercombo @0;
      navigation @1;
      vision @2;
      policy @3;
      offPolicy @4;
      onPolicy @5;
      chunked @6;
    }
  }

  enum Runner {
    snpe @0;
    tinygrad @1;
    stock @2;
  }

  struct Override {
    key @0 :Text;
    value @1 :Text;
  }

  struct ModelBundle {
    index @0 :UInt32;
    internalName @1 :Text;
    displayName @2 :Text;
    models @3 :List(Model);
    status @4 :DownloadStatus;
    generation @5 :UInt32;
    environment @6 :Text;
    runner @7 :Runner;
    is20hz @8 :Bool;
    ref @9 :Text;
    minimumSelectorVersion @10 :UInt32;
    overrides @11 :List(Override);
  }
}

struct LongitudinalPlanSP @0xf35cc4560bbf6ec2 {
  longitudinalPlanSource @1 :LongitudinalPlanSource;
  smartCruiseControl @2 :SmartCruiseControl;
  speedLimit @3 :SpeedLimit;
  vTarget @4 :Float32;
  aTarget @5 :Float32;
  events @6 :List(OnroadEventSP.Event);
  e2eAlerts @7 :E2eAlerts;

  deprecated :group {
    dec @0 :DynamicExperimentalControlDEPRECATED;
  }

  struct DynamicExperimentalControlDEPRECATED {
    state @0 :DynamicExperimentalControlState;
    enabled @1 :Bool;
    active @2 :Bool;

    enum DynamicExperimentalControlState {
      acc @0;
      blended @1;
    }
  }

  struct SmartCruiseControl {
    vision @0 :Vision;
    map @1 :Map;

    struct Vision {
      state @0 :VisionState;
      vTarget @1 :Float32;
      aTarget @2 :Float32;
      currentLateralAccel @3 :Float32;
      maxPredictedLateralAccel @4 :Float32;
      enabled @5 :Bool;
      active @6 :Bool;
    }

    struct Map {
      state @0 :MapState;
      vTarget @1 :Float32;
      aTarget @2 :Float32;
      enabled @3 :Bool;
      active @4 :Bool;
    }

    enum VisionState {
      disabled @0; # System disabled or inactive.
      enabled @1; # No predicted substantial turn on vision range.
      entering @2; # A substantial turn is predicted ahead, adapting speed to turn comfort levels.
      turning @3; # Actively turning. Managing acceleration to provide a roll on turn feeling.
      leaving @4; # Road ahead straightens. Start to allow positive acceleration.
      overriding @5; # System overriding with manual control.
    }

    enum MapState {
      disabled @0; # System disabled or inactive.
      enabled @1; # No predicted substantial turn on map range.
      turning @2; # Actively turning. Managing acceleration to provide a roll on turn feeling.
      overriding @3; # System overriding with manual control.
    }
  }

  struct SpeedLimit {
    resolver @0 :Resolver;
    assist @1 :Assist;

    struct Resolver {
      speedLimit @0 :Float32;
      distToSpeedLimit @1 :Float32;
      source @2 :Source;
      speedLimitOffset @3 :Float32;
      speedLimitLast @4 :Float32;
      speedLimitFinal @5 :Float32;
      speedLimitFinalLast @6 :Float32;
      speedLimitValid @7 :Bool;
      speedLimitLastValid @8 :Bool;
    }

    struct Assist {
      state @0 :AssistState;
      enabled @1 :Bool;
      active @2 :Bool;
      vTarget @3 :Float32;
      aTarget @4 :Float32;
    }

    enum Source {
      none @0;
      car @1;
      map @2;
    }

    enum AssistState {
      disabled @0;
      inactive @1; # No speed limit set or not enabled by parameter.
      preActive @2;
      pending @3; # Awaiting new speed limit.
      adapting @4; # Reducing speed to match new speed limit.
      active @5; # Cruising at speed limit.
    }
  }

  enum LongitudinalPlanSource {
    cruise @0;
    sccVision @1;
    sccMap @2;
    speedLimitAssist @3;
  }

  struct E2eAlerts {
    greenLightAlert @0 :Bool;
    leadDepartAlert @1 :Bool;
  }
}

struct OnroadEventSP @0xda96579883444c35 {
  events @0 :List(Event);

  struct Event {
    name @0 :EventName;

    # event types
    enable @1 :Bool;
    noEntry @2 :Bool;
    warning @3 :Bool;   # alerts presented only when  enabled or soft disabling
    userDisable @4 :Bool;
    softDisable @5 :Bool;
    immediateDisable @6 :Bool;
    preEnable @7 :Bool;
    permanent @8 :Bool; # alerts presented regardless of openpilot state
    overrideLateral @10 :Bool;
    overrideLongitudinal @9 :Bool;
  }

  enum EventName {
    lkasEnable @0;
    lkasDisable @1;
    manualSteeringRequired @2;
    manualLongitudinalRequired @3;
    silentLkasEnable @4;
    silentLkasDisable @5;
    silentBrakeHold @6;
    silentWrongGear @7;
    silentReverseGear @8;
    silentDoorOpen @9;
    silentSeatbeltNotLatched @10;
    silentParkBrake @11;
    controlsMismatchLateral @12;
    hyundaiRadarTracksConfirmed @13;
    experimentalModeSwitchedDEPRECATED @14;
    wrongCarModeAlertOnly @15;
    pedalPressedAlertOnly @16;
    laneTurnLeft @17;
    laneTurnRight @18;
    speedLimitPreActive @19;
    speedLimitActive @20;
    speedLimitChanged @21;
    speedLimitPending @22;
    e2eChime @23;
    laneChangeRoadEdge @24;
    bigModelReady @25;
  }
}

struct CarParamsSP @0x80ae746ee2596b11 {
  flags @0 :UInt32;        # flags for car specific quirks in sunnypilot
  safetyParam @1 : Int16;  # flags for sunnypilot's custom safety flags
  pcmCruiseSpeed @3 :Bool;
  intelligentCruiseButtonManagementAvailable @4 :Bool;
  enableGasInterceptor @5 :Bool;

  neuralNetworkLateralControl @2 :NeuralNetworkLateralControl;

  struct NeuralNetworkLateralControl {
    model @0 :Model;
    fuzzyFingerprint @1 :Bool;

    struct Model {
      path @0 :Text;
      name @1 :Text;
    }
  }
}

struct CarControlSP @0xa5cd762cd951a455 {
  mads @0 :ModularAssistiveDrivingSystem;
  params @1 :List(Param);
  leadOne @2 :LeadData;
  leadTwo @3 :LeadData;
  intelligentCruiseButtonManagement @4 :IntelligentCruiseButtonManagement;

  struct Param {
    key @0 :Text;
    type @2 :ParamType;
    value @3 :Data;

    valueDEPRECATED @1 :Text; # The data type change may cause issues with backwards compatibility.
  }

  enum ParamType {
    string @0;
    bool @1;
    int @2;
    float @3;
    time @4;
    json @5;
    bytes @6;
  }
}

struct BackupManagerSP @0xf98d843bfd7004a3 {
  backupStatus @0 :Status;
  restoreStatus @1 :Status;
  backupProgress @2 :Float32;
  restoreProgress @3 :Float32;
  lastError @4 :Text;
  currentBackup @5 :BackupInfo;
  backupHistory @6 :List(BackupInfo);

  enum Status {
    idle @0;
    inProgress @1;
    completed @2;
    failed @3;
  }

  struct Version {
    major @0 :UInt16;
    minor @1 :UInt16;
    patch @2 :UInt16;
    build @3 :UInt16;
    branch @4 :Text;
  }

  struct MetadataEntry {
    key @0 :Text;
    value @1 :Text;
    tags @2 :List(Text);
  }

  struct BackupInfo {
    deviceId @0 :Text;
    version @1 :UInt32;
    config @2 :Text;
    isEncrypted @3 :Bool;
    createdAt @4 :Text;  # ISO timestamp
    updatedAt @5 :Text;  # ISO timestamp
    sunnypilotVersion @6 :Version;
    backupMetadata @7 :List(MetadataEntry);
  }
}

struct CarStateSP @0xb86e6369214c01c8 {
  speedLimit @0 :Float32;
}

struct LiveMapDataSP @0xf416ec09499d9d19 {
  speedLimitValid @0 :Bool;
  speedLimit @1 :Float32;
  speedLimitAheadValid @2 :Bool;
  speedLimitAhead @3 :Float32;
  speedLimitAheadDistance @4 :Float32;
  roadName @5 :Text;
}

struct ModelDataV2SP @0xa1680744031fdb2d {
  laneTurnDirection @0 :TurnDirection;
  leftLaneChangeEdgeBlock @1 :Bool;
  rightLaneChangeEdgeBlock @2 :Bool;
  blendfulCurvature @3 :Float32;
  blendfulTrust @4 :Float32;
  modelCurvature @5 :Float32;
  blendfulGoingAround @6 :Bool;
  blendfulLeftProb @7 :Float32;
  blendfulRightProb @8 :Float32;
  blendfulLaneWidth @9 :Float32;


  enum TurnDirection {
    none @0;
    turnLeft @1;
    turnRight @2;
  }
}

struct CustomReserved10 @0xcb9fd56c7057593a {
}

struct CustomReserved11 @0xc2243c65e0340384 {
}

struct CustomReserved12 @0x9ccdc8676701b412 {
}

struct CustomReserved13 @0xcd96dafb67a082d0 {
}

struct CustomReserved14 @0xb057204d7deadf3f {
}

struct CustomReserved15 @0xbd443b539493bc68 {
}

struct CustomReserved16 @0xfc6241ed8877b611 {
}

struct CustomReserved17 @0xa30662f84033036c {
}

struct CustomReserved18 @0xc86a3d38d13eb3ef {
}

struct BlendfulInster @0xa4f1eb3323f5f582 {
  # Right-line Blendful (A8), its own message (rev 7.1: moved out of ModelDataV2SP, renumbered) (implementation round 3: renumbered without the fields of the never-driven first implementation).
  # Lists named "input" hold the four inputs in the order [left line, left edge, right line, right edge]; position lists hold
  # x = 0, 0.5 L, L, 1.5 L per input (L = max(1.5 v, 10 m)).
  blendfulMode @0 :UInt8;  # 0 off, 1 regular, 2 right line
  blendfulRightApplied @1 :Bool;
  blendfulRightGap @2 :Float32;  # g(v), m
  blendfulRightGapNow @3 :Float32;  # right wheel to the right reference now, m
  blendfulRightQuality @4 :Float32;  # q_R
  blendfulLeftRefQuality @5 :Float32;
  blendfulAuthority @6 :Float32;  # a
  blendfulAuthorityTarget @7 :Float32;  # a*
  blendfulFadeBend @8 :Float32;
  blendfulFadeYieldLeft @9 :Float32;
  blendfulFadeYieldRight @10 :Float32;
  blendfulFadeJump @11 :Float32;
  blendfulFadePlausibility @12 :Float32;
  blendfulFadeBlinker @13 :Float32;
  blendfulAwayCar @14 :Float32;  # rev 7.1: the detector's swerve value |offset - usual| - curve allowance, m (was the endpoint gap)
  blendfulStrength @15 :Float32;  # s_R
  blendfulStrengthWeight @16 :Float32;  # w_c
  blendfulRightOnly @17 :Bool;  # right-only episode: q_R from line or edge with left line quality < 0.3
  blendfulCurveAccel @18 :Float32;  # |kappa_m| v^2, m/s^2
  blendfulYieldActive @19 :Bool;
  blendfulCapActive @20 :Bool;
  blendfulBoundActive @21 :Bool;
  blendfulKappaUp @22 :Float32;  # reference instance output (instantaneous)
  blendfulKappaRL @23 :Float32;  # right-line part, profile amplitude
  blendfulKappa0 @24 :Float32;  # delivered profile amplitude
  blendfulGainF @25 :Float32;
  blendfulProfileLength @26 :Float32;  # L_c, m
  blendfulProfileShift @27 :Float32;  # D, m
  blendfulFilterTime @28 :Float32;  # ms per frame, the four input filters only
  blendfulInputFiltered @29 :List(Float32);
  blendfulInputRaw @30 :List(Float32);
  blendfulFilterOffset @31 :List(Float32);  # filtered minus raw at x = L, per input
  blendfulInputConfidence @32 :List(Float32);
  blendfulInputState @33 :List(UInt8);  # bit 0 coasting, bit 1 re-initialised this frame
  blendfulCandidateConfidence @34 :List(Float32);  # c_i per input (hysteresis gate)
  blendfulCandidateWeight @35 :List(Float32);  # softmin weight w_i per input at x = L
  blendfulReference @36 :List(Float32);  # [left, right] reference at x = L
  blendfulTarget @37 :List(Float32);  # target lateral position per model x index (right line mode)
  blendfulJunctionEpisode @38 :Bool;
  blendfulJunctionCues @39 :UInt8;  # bit 0 widening in frame, 1 widening over time, 2 edge divergence, 3 gore / kink, 4 sharp bend
  blendfulJunctionCueValues @40 :List(Float32);  # [widening in frame m, width excess m, edge divergence m, edge angle deg, gore min separation m, kink deg, bend b]
  blendfulJunctionEpisodeDistance @41 :Float32;  # m since the episode started
  blendfulJunctionWidening @42 :Float32;  # the widening quantity of the episode end rule, m
  blendfulWidthMemory @43 :Float32;  # m (0 while not armed)
  blendfulWidthMemoryAge @44 :Float32;  # s since the last update
  blendfulWidthMemoryState @45 :UInt8;  # bit 0 armed, 1 stale, 2 measuring
  blendfulModelAloneTime @46 :Float32;  # s spent in junction episodes since the mode started
  blendfulJunctionEnded @47 :UInt8;  # 0 no end this frame, 1 quiet (2 s and 30 m), 2 widening gone (1 s), 3 maximum length (300 m)
  blendfulRoadClass @48 :UInt8;  # 0 edges only, 1 shared, 2 lanes, 3 motorway (hysteresis, 5 s dwell)
  blendfulFrameDt @49 :Float32;  # true model frame interval used by the filters and the target, s
  blendfulPlanSpeed @50 :Float32;  # the raw plan's own v_x(0), m/s
  blendfulTargetLag @51 :Float32;  # raw target minus rate-limited target at x = L, m
  blendfulDClipped @52 :Bool;  # the fitted amplitude was clipped at 1.0 m
  blendfulYieldSide @53 :Int8;  # capped obstacle side(s) while a no-growth phase runs: +1 right, -1 left, 2 both (side not yet known, or triggers disagree), 0 none
  blendfulYieldSource @54 :UInt8;  # reference of the endpoint-gap trigger: 0 plan alone, 1 right, 2 left
  blendfulYieldMoving @55 :Bool;  # unused since fix 1 (always false); was: a move to the right is on (running minimum of the endpoint gap)
  blendfulCapMemory @56 :Float32;  # toward-obstacle correction held at the no-growth point (1/m)
  blendfulUpdateTime @57 :Float32;  # ms, the whole Blendful update of this frame (reference, right-line instance, cap, limiter); 0 when it did not run
  blendfulNonFinite @58 :Bool;  # a non-finite input or result this frame: released toward 0, instances reset
  blendfulCentreSource @59 :UInt8;  # left-limit form: 0 midpoint of both references, 1 right reference, 2 left reference (0.5 / 0.4, 0.5 s dwell)
  # rev 6.5 final move-speed limit (B5 step 4b): values at x = L
  blendfulApproach @60 :Float32;  # weighted approach A to the right reference since the last reset, m
  blendfulGapReset @61 :Float32;  # G0, the gap to the right reference at the last reset (rebased by reference switches), m
  blendfulMoveLimiting @62 :Bool;  # the 0.05 m/s limit binds somewhere in the window
  blendfulMoveReset @63 :UInt8;  # reset reasons this frame: bit 0 a = 0, 1 junction end, 2 yield end, 3 right reference regained, 4 mode on, 5 limiter created, 6 w <= 0
  blendfulSwitchPart @64 :Float32;  # switch part of the change of Y_R added to G0 this frame, m
  blendfulRotationScale @65 :UInt8;  # FILTER_ROTATION_SCALE: 0 design, 1 model_axis
  # radar right-side clearance check (B6a item 8)
  blendfulRadarHealthy @66 :Bool;  # f_radar = 1 (tracks fresh, no error flags, 1 s since they returned)
  blendfulRadarObjects @67 :UInt8;  # confirmed objects in the right corridor
  blendfulRadarMemory @68 :UInt8;  # of them remembered beside the car
  blendfulRadarMoving @69 :UInt8;  # of them classified moving (log only)
  blendfulRadarPopups @70 :UInt8;  # confirmed this frame less than 2 s ahead (log only)
  blendfulRadarNearest @71 :List(Float32);  # [x, y, clearance of the car's right side] of the corridor object with the largest deficit, m
  blendfulObjectBound @72 :Float32;  # held object bound T_obj (1000 = none)
  blendfulObjectBoundHold @73 :Float32;  # s since the object bound last fell
  blendfulObjectBoundActive @74 :Bool;
  blendfulClearanceDeficit @75 :Float32;  # d = m(v) - clearance at the centre limit, m (> 0 starts the clearance yield)
  blendfulClearanceYield @76 :Bool;
  blendfulClearanceMargin @77 :Float32;  # m(v) = g(v) + e(v), m
  blendfulRightBlindspot @78 :Bool;
  # rev 6.10 "Start and floor"
  blendfulDeliveredSource @79 :UInt8;  # right-line share delivered: 0 right-line correction, 1 Regular's (larger of, target left of the centre), 2 right-line (larger of)
  blendfulTargetMinusCentre @80 :Float32;  # T_RL - T_C' at x = L, m (< 0: the right-line target is left of the lane centre)
  # rev 6.12
  blendfulAuthorityLimited @81 :Bool;  # Regular delivered while T_RL is right of the centre for > 2 s (model pull; with blendfulDClipped, blendfulStrength)
  # hand-over to the model's path (owner 13 Nov 09:27): from the latch of Keep Right's own swerve trigger the whole correction goes to 0
  blendfulHandOver @82 :Bool;  # the hand-over is active this frame
  blendfulHandOverCorrection @83 :Float32;  # delivered correction c at the latch of the last hand-over, 1/m
  blendfulHandOverTime @84 :Float32;  # s from the latch of the last hand-over until c reached 0, or until the hand-over ended if earlier (then kept)
  # rev 7 / 7.1: strength filter and the single detector
  blendfulStrengthFromConfidence @85 :Float32;  # a*_q = clip((q_R - 0.2) / 0.5, 0, 1) before the factors
  blendfulDetectorOffset @86 :Float32;  # mean (plan_y - lane centre) over 1.4-1.6 s, our own estimated move subtracted, m
  blendfulDetectorUsual @87 :Float32;  # 10 s (20 s going around) low-pass of the offset, m
  blendfulDetectorOwnMove @88 :Float32;  # estimated sideways move from our correction (30 s high-pass), m; (1 - 0.35) of it is subtracted
  blendfulDetectorAway @89 :Float32;  # swerve minus the car-departure term, m (latch > 0.2)
}

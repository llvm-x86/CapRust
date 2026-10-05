//! Audio envelope overlay for timeline clips (issue #14, v1: read-only).
//!
//! Two overlays share one vertical scale:
//! - the clip's volume automation (keyframe polyline, or a flat line at
//!   `volume_db` when there are no keyframes);
//! - ducking zones: where the clip's `duck_against` control clip overlaps it.
//!
//! The module has two layers. The shape is computed in clip-normalised
//! space (x = fraction of the clip duration, y = fraction of the clip
//! height measured from the top) by plain functions with no egui types, so
//! zoom and scroll never invalidate it and the math is unit-tested below.
//! `draw_audio_envelope` only maps that shape to pixels.
//!
//! Keyframe semantics mirror `RenderPlan::build_audio_chain` in
//! `caprust-media-io::export_graph`, which is what both preview and export
//! render:
//! - `VolumeKeyframe::t_ms` is clip-local timeline time (the `volume`
//!   filter runs after trim, `asetpts=PTS-STARTPTS` and `atempo`), so it is
//!   independent of `source_offset_ms` and `speed`;
//! - keyframes are stably sorted by `t_ms`, the gain is held before the
//!   first and after the last, and linear in dB in between; a non-empty
//!   curve replaces `volume_db` entirely;
//! - the overlay shows the clip's own envelope only. Fades, the track
//!   fader and normalize/denoise stages are not part of it.

use caprust_core::clip::VolumeKeyframe;
use caprust_core::{Clip, ClipType, ProjectState};
use egui::{pos2, Color32, Painter, Pos2, Rect, Shape, Stroke};
use std::collections::HashMap;
use uuid::Uuid;

use crate::theme::tokens::elev;

/// Gain drawn at the top edge of the clip. Anything louder is clamped.
pub const DB_TOP: f32 = 12.0;
/// Gain drawn at the bottom edge of the clip. Anything quieter is clamped.
pub const DB_BOTTOM: f32 = -60.0;
/// Where 0 dB sits, as a fraction of the clip height measured from the top
/// (0.4 from the top = 60 % of the height up from the bottom). Issue #14
/// words this as "0 dB at ~60% height"; read bottom-up, as for a fader.
pub const UNITY_Y: f32 = 0.4;

/// Duck-zone fill alpha range. `duck_reduction_db` 0 maps to the minimum,
/// the renderer's floor of -60 dB to the maximum.
const DUCK_ALPHA_MIN: f32 = 28.0;
const DUCK_ALPHA_MAX: f32 = 120.0;

/// Vertical padding so the line at the extremes stays inside the border.
const EDGE_PAD: f32 = 3.0;

/// dB -> fraction of the clip height from the top: +12 dB = 0.0,
/// 0 dB = [`UNITY_Y`], -60 dB = 1.0. Piecewise linear, clamped outside
/// [`DB_BOTTOM`, `DB_TOP`]. NaN is drawn at unity.
pub fn db_to_y_frac(db: f32) -> f32 {
    if db.is_nan() {
        return UNITY_Y;
    }
    let db = db.clamp(DB_BOTTOM, DB_TOP);
    if db >= 0.0 {
        UNITY_Y * (1.0 - db / DB_TOP)
    } else {
        UNITY_Y + (1.0 - UNITY_Y) * (db / DB_BOTTOM)
    }
}

/// Non-finite gains cannot come from a well-formed project file but can
/// from a script or a hand-edited one; keep the math finite.
fn sane_db(db: f32) -> f64 {
    if db.is_nan() {
        0.0
    } else {
        db.clamp(-1000.0, 1000.0) as f64
    }
}

/// Gain in dB at clip-local time `t` (ms) for keyframes sorted by time.
/// Same evaluation order as the nested `if(lt(t,..))` expression the
/// renderer emits: duplicate times make a step, not a divide-by-zero.
fn value_at(kfs: &[(f64, f64)], t: f64) -> f64 {
    if t < kfs[0].0 {
        return kfs[0].1;
    }
    for w in kfs.windows(2) {
        if t < w[1].0 {
            return w[0].1 + (w[1].1 - w[0].1) * (t - w[0].0) / (w[1].0 - w[0].0);
        }
    }
    kfs[kfs.len() - 1].1
}

/// Normalised polyline `[x, y]` for a clip's volume envelope, from x = 0
/// (clip head) to x = 1 (clip tail). Empty when `duration_ms` is 0.
///
/// Keyframes outside `0..duration_ms` do not get a vertex; the line is cut
/// at the clip edges at the interpolated gain, as the renderer cuts it.
/// Segments that cross +12 / 0 / -60 dB get a vertex at the crossing,
/// because the dB -> y scale has a corner at 0 dB and flat ends beyond the
/// clamps; without it a long ramp would be drawn as one wrong straight line.
pub fn volume_line(
    keyframes: &[VolumeKeyframe],
    static_db: f32,
    duration_ms: u64,
) -> Vec<[f32; 2]> {
    if duration_ms == 0 {
        return Vec::new();
    }
    let dur = duration_ms as f64;

    let mut sorted: Vec<&VolumeKeyframe> = keyframes.iter().collect();
    sorted.sort_by_key(|k| k.t_ms); // stable, like the renderer
    let kfs: Vec<(f64, f64)> = sorted
        .iter()
        .map(|k| (k.t_ms as f64, sane_db(k.gain_db)))
        .collect();

    let mut bp: Vec<(f64, f64)> = Vec::with_capacity(kfs.len() + 2);
    if kfs.is_empty() {
        let g = sane_db(static_db);
        bp.extend([(0.0, g), (dur, g)]);
    } else {
        bp.push((0.0, value_at(&kfs, 0.0)));
        bp.extend(kfs.iter().copied().filter(|&(t, _)| t > 0.0 && t < dur));
        bp.push((dur, value_at(&kfs, dur)));
    }

    let point = |t: f64, g: f64| [(t / dur) as f32, db_to_y_frac(g as f32)];
    let mut out = Vec::with_capacity(bp.len() + 4);
    out.push(point(bp[0].0, bp[0].1));
    for w in bp.windows(2) {
        let ((t0, g0), (t1, g1)) = (w[0], w[1]);
        if t1 > t0 && g1 != g0 {
            // Visit the thresholds in the order the segment crosses them.
            let ths = if g1 > g0 {
                [DB_BOTTOM, 0.0, DB_TOP]
            } else {
                [DB_TOP, 0.0, DB_BOTTOM]
            };
            for th in ths {
                let f = (th as f64 - g0) / (g1 - g0);
                if f > 0.0 && f < 1.0 {
                    out.push(point(t0 + f * (t1 - t0), th as f64));
                }
            }
        }
        out.push(point(t1, g1));
    }
    out
}

/// Overlap of a clip's timeline span with its control clip's span, as
/// clip-local `(start_ms, end_ms)`. `None` when they do not overlap
/// (touching edges do not count).
pub fn duck_span(
    clip_start: u64,
    clip_dur: u64,
    ctrl_start: u64,
    ctrl_dur: u64,
) -> Option<(u64, u64)> {
    let start = clip_start.max(ctrl_start);
    let end = clip_start
        .saturating_add(clip_dur)
        .min(ctrl_start.saturating_add(ctrl_dur));
    (end > start).then(|| (start - clip_start, end - clip_start))
}

/// Duck-zone fill alpha for a clip's `duck_reduction_db`. The renderer
/// clamps the value to [-60, 0] dB; deeper ducking is drawn more opaque.
pub fn duck_alpha(reduction_db: f32) -> u8 {
    let reduction_db = if reduction_db.is_nan() {
        0.0
    } else {
        reduction_db
    };
    let depth = (-reduction_db).clamp(0.0, 60.0) / 60.0;
    (DUCK_ALPHA_MIN + depth * (DUCK_ALPHA_MAX - DUCK_ALPHA_MIN)).round() as u8
}

/// The clip that actually drives this clip's ducking, or `None`. The mix
/// only builds a sidechain when the control id resolves to a clip that
/// contributes audio (not self, not on a muted track, not a detached video
/// or a clip type without audio); anything else leaves the clip un-ducked.
fn duck_control<'a>(project: &'a ProjectState, clip: &Clip) -> Option<&'a Clip> {
    let id = clip.duck_against.filter(|&id| id != clip.id)?;
    let ctrl = project.clips.iter().find(|c| c.id == id)?;
    let has_audio = match ctrl.clip_type {
        ClipType::Audio { .. } | ClipType::Narration { .. } => true,
        ClipType::Video { .. } => !ctrl.audio_detached,
        _ => false,
    };
    let track_live = project
        .tracks
        .get(ctrl.track_index)
        .is_some_and(|t| !t.muted);
    (has_audio && track_live).then_some(ctrl)
}

/// Shaded span where a clip is ducked, as fractions of the clip duration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DuckZone {
    pub x0: f32,
    pub x1: f32,
    pub alpha: u8,
}

/// Everything the overlay draws for one clip, in normalised space.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipEnvelope {
    pub line: Vec<[f32; 2]>,
    pub duck: Option<DuckZone>,
}

impl ClipEnvelope {
    /// The zone is the plain span overlap with the control clip, i.e. where
    /// the clip is meant to duck. `sidechaincompress` is signal driven
    /// (threshold 0.05, release 500 ms), so pauses inside the control clip
    /// and the release tail after it are not modelled.
    pub fn build(project: &ProjectState, clip: &Clip) -> Self {
        let dur = clip.duration_ms.max(1) as f32;
        let duck = duck_control(project, clip).and_then(|ctrl| {
            duck_span(
                clip.start_time_ms,
                clip.duration_ms,
                ctrl.start_time_ms,
                ctrl.duration_ms,
            )
            .map(|(a, b)| DuckZone {
                x0: a as f32 / dur,
                x1: b as f32 / dur,
                alpha: duck_alpha(clip.duck_reduction_db),
            })
        });
        Self {
            line: volume_line(&clip.volume_keyframes, clip.volume_db, clip.duration_ms),
            duck,
        }
    }
}

/// Per-clip envelope cache. Entries are valid for one `render_hash()`:
/// it covers every input of [`ClipEnvelope::build`] (keyframes, volume,
/// duck settings, clip and control spans, track mute), so any edit drops
/// the whole cache. Zoom, scroll and rect never do: they only enter at
/// draw time.
#[derive(Default)]
pub struct EnvelopeCache {
    hash: u64,
    by_clip: HashMap<Uuid, ClipEnvelope>,
}

impl EnvelopeCache {
    pub fn get(&mut self, render_hash: u64, project: &ProjectState, clip: &Clip) -> &ClipEnvelope {
        if self.hash != render_hash {
            self.by_clip.clear();
            self.hash = render_hash;
        }
        self.by_clip
            .entry(clip.id)
            .or_insert_with(|| ClipEnvelope::build(project, clip))
    }
}

/// Draw the overlay on top of a clip's waveform. `rect` is the clip's full
/// (unclipped) rect; `painter` must already be clipped to the visible part
/// of the clip. The duck zone is drawn first so the line stays on top.
pub fn draw_audio_envelope(painter: &Painter, rect: Rect, env: &ClipEnvelope, color: Color32) {
    if let Some(z) = &env.duck {
        let zone = Rect::from_min_max(
            pos2(rect.left() + z.x0 * rect.width(), rect.top()),
            pos2(rect.left() + z.x1 * rect.width(), rect.bottom()),
        );
        let [r, g, b, _] = color.to_array();
        painter.rect_filled(zone, 0.0, Color32::from_rgba_unmultiplied(r, g, b, z.alpha));
    }
    if env.line.len() >= 2 {
        let h = (rect.height() - 2.0 * EDGE_PAD).max(0.0);
        let pts: Vec<Pos2> = env
            .line
            .iter()
            .map(|&[x, y]| {
                pos2(
                    rect.left() + x * rect.width(),
                    rect.top() + EDGE_PAD + y * h,
                )
            })
            .collect();
        // Dark halo first: the line shares the waveform's hue.
        painter.add(Shape::line(
            pts.clone(),
            Stroke::new(elev::STROKE_EMPHASIS + 2.5, Color32::from_black_alpha(150)),
        ));
        painter.add(Shape::line(pts, Stroke::new(elev::STROKE_EMPHASIS, color)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kf(t_ms: u64, gain_db: f32) -> VolumeKeyframe {
        VolumeKeyframe { t_ms, gain_db }
    }

    fn near(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    fn assert_line(got: &[[f32; 2]], want: &[[f32; 2]]) {
        assert_eq!(got.len(), want.len(), "got {got:?}, want {want:?}");
        for (g, w) in got.iter().zip(want) {
            assert!(
                near(g[0], w[0]) && near(g[1], w[1]),
                "got {got:?}, want {want:?}"
            );
        }
    }

    // ---- scale ----

    #[test]
    fn scale_anchors_and_clamps() {
        assert!(near(db_to_y_frac(12.0), 0.0));
        assert!(near(db_to_y_frac(0.0), UNITY_Y));
        assert!(near(db_to_y_frac(-60.0), 1.0));
        // Midpoints of each linear half.
        assert!(near(db_to_y_frac(6.0), UNITY_Y / 2.0));
        assert!(near(db_to_y_frac(-30.0), UNITY_Y + (1.0 - UNITY_Y) / 2.0));
        // Beyond the range clamps instead of leaving the clip rect.
        assert!(near(db_to_y_frac(40.0), 0.0));
        assert!(near(db_to_y_frac(-200.0), 1.0));
        assert!(near(db_to_y_frac(f32::INFINITY), 0.0));
        assert!(near(db_to_y_frac(f32::NEG_INFINITY), 1.0));
        assert!(near(db_to_y_frac(f32::NAN), UNITY_Y));
    }

    #[test]
    fn louder_is_always_higher() {
        let mut prev = f32::MAX;
        for db in (-60..=12).map(|d| d as f32) {
            let y = db_to_y_frac(db);
            assert!(y < prev || (db == -60.0), "y must decrease as dB rises");
            prev = y;
        }
    }

    // ---- volume line ----

    #[test]
    fn no_keyframes_is_flat_at_static_volume() {
        let y = db_to_y_frac(-6.0);
        assert_line(&volume_line(&[], -6.0, 4000), &[[0.0, y], [1.0, y]]);
        assert_line(
            &volume_line(&[], 0.0, 4000),
            &[[0.0, UNITY_Y], [1.0, UNITY_Y]],
        );
    }

    #[test]
    fn zero_duration_draws_nothing() {
        assert!(volume_line(&[], 0.0, 0).is_empty());
        assert!(volume_line(&[kf(0, 3.0)], 0.0, 0).is_empty());
    }

    #[test]
    fn keyframes_replace_static_volume() {
        // volume_db = -30 must be ignored once a curve exists.
        let line = volume_line(&[kf(0, 0.0)], -30.0, 1000);
        assert_line(&line, &[[0.0, UNITY_Y], [1.0, UNITY_Y]]);
    }

    #[test]
    fn single_keyframe_holds_for_whole_clip() {
        // Held before and after the lone keyframe, wherever it sits.
        let y = db_to_y_frac(-12.0);
        assert_line(
            &volume_line(&[kf(500, -12.0)], 0.0, 1000),
            &[[0.0, y], [0.5, y], [1.0, y]],
        );
    }

    #[test]
    fn two_keyframes_hold_then_ramp_then_hold() {
        // Keyframes at 25 % and 75 %; held at the first gain before, at
        // the last after (flat ends), linear in between.
        let a = db_to_y_frac(-6.0);
        let b = db_to_y_frac(-24.0);
        let line = volume_line(&[kf(250, -6.0), kf(750, -24.0)], 0.0, 1000);
        assert_line(&line, &[[0.0, a], [0.25, a], [0.75, b], [1.0, b]]);
    }

    #[test]
    fn unsorted_keyframes_are_sorted_like_the_renderer() {
        let sorted = volume_line(&[kf(250, -6.0), kf(750, -24.0)], 0.0, 1000);
        let shuffled = volume_line(&[kf(750, -24.0), kf(250, -6.0)], 0.0, 1000);
        assert_line(&shuffled, &sorted);
    }

    #[test]
    fn keyframes_at_clip_edges_add_no_extra_vertices() {
        let a = db_to_y_frac(-3.0);
        let b = db_to_y_frac(-9.0);
        let line = volume_line(&[kf(0, -3.0), kf(1000, -9.0)], 0.0, 1000);
        assert_line(&line, &[[0.0, a], [1.0, b]]);
    }

    #[test]
    fn keyframes_past_the_clip_end_cut_at_the_interpolated_gain() {
        // Trimmed clip: 1000 ms long, curve was authored for 2000 ms.
        // At t = 1000 the ramp 0 dB -> -12 dB (0..2000) is at -6 dB.
        let line = volume_line(&[kf(0, 0.0), kf(2000, -12.0)], 0.0, 1000);
        assert_line(&line, &[[0.0, UNITY_Y], [1.0, db_to_y_frac(-6.0)]]);
        // All keyframes beyond the end: the first one holds backwards.
        let line = volume_line(&[kf(3000, -6.0), kf(4000, -12.0)], 0.0, 1000);
        let y = db_to_y_frac(-6.0);
        assert_line(&line, &[[0.0, y], [1.0, y]]);
    }

    #[test]
    fn keyframes_are_clip_local_timeline_time() {
        // The renderer's `volume` filter runs after trim and atempo, so
        // keyframe time is measured from the clip head on the timeline:
        // a split-off right half (source_offset_ms > 0) or a sped-up clip
        // draws the same line as an untouched one.
        let mut plain = Clip::new_audio("a.wav", A1, 10_000, 2000);
        plain.volume_keyframes = vec![kf(500, 0.0), kf(1500, -12.0)];
        let mut moved = plain.clone();
        moved.source_offset_ms = 5000;
        moved.speed = 2.0;
        let p = project_with(vec![]);
        let want = ClipEnvelope::build(&p, &plain).line;
        assert_line(
            &want,
            &[
                [0.0, UNITY_Y],
                [0.25, UNITY_Y],
                [0.75, db_to_y_frac(-12.0)],
                [1.0, db_to_y_frac(-12.0)],
            ],
        );
        assert_eq!(ClipEnvelope::build(&p, &moved).line, want);
    }

    #[test]
    fn gains_beyond_the_scale_clamp_flat() {
        let top = volume_line(&[kf(0, 30.0), kf(1000, 20.0)], 0.0, 1000);
        assert_line(&top, &[[0.0, 0.0], [1.0, 0.0]]);
        let bottom = volume_line(&[kf(0, -90.0), kf(1000, -200.0)], 0.0, 1000);
        assert_line(&bottom, &[[0.0, 1.0], [1.0, 1.0]]);
    }

    #[test]
    fn ramp_through_the_scale_corners_gets_crossing_vertices() {
        // +24 dB -> -72 dB over 960 ms: the line leaves the +12 clamp at
        // 1/8 of the ramp, passes 0 dB at 1/4, and hits -60 at 7/8.
        let line = volume_line(&[kf(0, 24.0), kf(960, -72.0)], 0.0, 960);
        assert_line(
            &line,
            &[
                [0.0, 0.0],
                [0.125, 0.0],
                [0.25, UNITY_Y],
                [0.875, 1.0],
                [1.0, 1.0],
            ],
        );
    }

    #[test]
    fn rising_ramp_crosses_in_ascending_order() {
        let line = volume_line(&[kf(0, -60.0), kf(600, 12.0)], 0.0, 600);
        assert_line(&line, &[[0.0, 1.0], [5.0 / 6.0, UNITY_Y], [1.0, 0.0]]);
    }

    #[test]
    fn duplicate_keyframe_times_make_a_step() {
        // Renderer: before t=500 -> -6, from t=500 -> -18.
        let line = volume_line(&[kf(500, -6.0), kf(500, -18.0)], 0.0, 1000);
        let a = db_to_y_frac(-6.0);
        let b = db_to_y_frac(-18.0);
        assert_line(&line, &[[0.0, a], [0.5, a], [0.5, b], [1.0, b]]);
    }

    #[test]
    fn non_finite_gains_stay_finite() {
        let line = volume_line(&[kf(0, f32::NAN), kf(500, f32::INFINITY)], 0.0, 1000);
        assert!(line
            .iter()
            .flatten()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
    }

    // ---- duck span / alpha ----

    #[test]
    fn duck_span_overlaps() {
        // Clip 1000..5000, control 3000..8000 -> clip-local 2000..4000.
        assert_eq!(duck_span(1000, 4000, 3000, 5000), Some((2000, 4000)));
        // Control fully inside the clip.
        assert_eq!(duck_span(0, 10_000, 2000, 1000), Some((2000, 3000)));
        // Control covers the whole clip.
        assert_eq!(duck_span(2000, 1000, 0, 10_000), Some((0, 1000)));
        // Control starts before, ends inside.
        assert_eq!(duck_span(1000, 4000, 0, 2000), Some((0, 1000)));
    }

    #[test]
    fn duck_span_disjoint_or_touching_is_none() {
        assert_eq!(duck_span(0, 1000, 2000, 1000), None);
        assert_eq!(duck_span(2000, 1000, 0, 1000), None);
        assert_eq!(duck_span(0, 1000, 1000, 1000), None, "touching edges");
        assert_eq!(duck_span(0, 0, 0, 1000), None, "empty clip");
        assert_eq!(duck_span(0, 1000, 500, 0), None, "empty control");
    }

    #[test]
    fn duck_span_survives_overflow() {
        assert_eq!(
            duck_span(u64::MAX - 10, 100, u64::MAX - 5, 100),
            Some((5, 10))
        );
    }

    #[test]
    fn deeper_ducking_is_more_opaque_and_clamped() {
        let a = |db| duck_alpha(db);
        assert_eq!(a(0.0), DUCK_ALPHA_MIN as u8);
        assert_eq!(a(-60.0), DUCK_ALPHA_MAX as u8);
        assert!(a(-6.0) < a(-12.0) && a(-12.0) < a(-24.0));
        assert_eq!(a(-500.0), a(-60.0), "renderer clamps at -60");
        assert_eq!(a(6.0), a(0.0), "boost is not ducking");
        assert_eq!(a(f32::NAN), a(0.0));
    }

    // ---- ClipEnvelope::build ----

    /// Index of the default "A1" audio track in `default_tracks()`.
    const A1: usize = 3;

    fn project_with(clips: Vec<Clip>) -> ProjectState {
        let mut p = ProjectState::default();
        assert_eq!(p.tracks[A1].kind, caprust_core::TrackKind::Audio);
        for c in clips {
            p.add_clip(c);
        }
        p
    }

    #[test]
    fn build_without_duck_has_no_zone() {
        let c = Clip::new_audio("music.wav", A1, 0, 4000);
        let p = project_with(vec![c.clone()]);
        let env = ClipEnvelope::build(&p, &c);
        assert_eq!(env.duck, None);
        assert_eq!(env.line.len(), 2);
    }

    #[test]
    fn build_duck_zone_from_control_span() {
        let narr = Clip::new_audio("voice.wav", A1, 2000, 1000);
        let mut music = Clip::new_audio("music.wav", A1, 0, 4000);
        music.duck_against = Some(narr.id);
        music.duck_reduction_db = -24.0;
        let p = project_with(vec![narr, music.clone()]);
        let z = ClipEnvelope::build(&p, &music).duck.expect("zone");
        assert!(near(z.x0, 0.5) && near(z.x1, 0.75));
        assert_eq!(z.alpha, duck_alpha(-24.0));
    }

    #[test]
    fn build_duck_zone_respects_trim_and_offset() {
        // Trimmed clip (starts 1000 ms into its source, placed at 10 s):
        // the zone is in timeline time, so source_offset_ms is irrelevant.
        let narr = Clip::new_audio("voice.wav", A1, 11_000, 500);
        let mut music = Clip::new_audio("music.wav", A1, 10_000, 2000);
        music.source_offset_ms = 1000;
        music.duck_against = Some(narr.id);
        let p = project_with(vec![narr, music.clone()]);
        let z = ClipEnvelope::build(&p, &music).duck.expect("zone");
        assert!(near(z.x0, 0.5) && near(z.x1, 0.75));
    }

    #[test]
    fn build_duck_ignores_missing_self_and_non_overlapping_control() {
        let ghost = Clip::new_audio("ghost.wav", A1, 0, 1000); // not in project
        let mut music = Clip::new_audio("music.wav", A1, 0, 4000);
        music.duck_against = Some(ghost.id);
        let p = project_with(vec![music.clone()]);
        assert_eq!(
            ClipEnvelope::build(&p, &music).duck,
            None,
            "missing control"
        );

        let mut selfie = Clip::new_audio("s.wav", A1, 0, 4000);
        selfie.duck_against = Some(selfie.id);
        let p = project_with(vec![selfie.clone()]);
        assert_eq!(
            ClipEnvelope::build(&p, &selfie).duck,
            None,
            "self reference"
        );

        let late = Clip::new_audio("late.wav", A1, 9000, 1000);
        let mut music = Clip::new_audio("music.wav", A1, 0, 4000);
        music.duck_against = Some(late.id);
        let p = project_with(vec![late, music.clone()]);
        assert_eq!(ClipEnvelope::build(&p, &music).duck, None, "no overlap");
    }

    #[test]
    fn build_duck_needs_a_control_that_contributes_audio() {
        // Mirrors plan_from_project: muted track, detached video audio and
        // silent clip types never enter the mix, so nothing ducks.
        let mut music = Clip::new_audio("music.wav", A1, 0, 4000);

        let narr = Clip::new_audio("voice.wav", A1, 0, 4000);
        music.duck_against = Some(narr.id);
        let mut p = project_with(vec![narr, music.clone()]);
        assert!(ClipEnvelope::build(&p, &music).duck.is_some());
        p.tracks[A1].muted = true;
        assert_eq!(ClipEnvelope::build(&p, &music).duck, None, "muted track");

        let mut vid = Clip::new_video("v.mp4", 1, 0, 4000);
        music.duck_against = Some(vid.id);
        vid.audio_detached = true;
        let mut p = project_with(vec![vid.clone(), music.clone()]);
        assert_eq!(ClipEnvelope::build(&p, &music).duck, None, "detached video");
        p.clips[0].audio_detached = false;
        assert!(
            ClipEnvelope::build(&p, &music).duck.is_some(),
            "video audio"
        );

        let img = Clip::new_image("i.png", 1, 0, 4000);
        music.duck_against = Some(img.id);
        let p = project_with(vec![img, music.clone()]);
        assert_eq!(ClipEnvelope::build(&p, &music).duck, None, "image clip");
    }

    // ---- drawing ----

    #[test]
    fn draw_maps_loud_to_the_top_and_offsets_by_the_full_rect() {
        // Full clip rect starts left of the visible area (scrolled) and is
        // 200 px wide; +12 dB must land at the top, -60 dB at the bottom.
        let rect = Rect::from_min_max(pos2(-50.0, 10.0), pos2(150.0, 70.0));
        let env = ClipEnvelope {
            line: volume_line(&[kf(0, 12.0), kf(1000, -60.0)], 0.0, 1000),
            duck: Some(DuckZone {
                x0: 0.25,
                x1: 0.5,
                alpha: 80,
            }),
        };
        let ctx = egui::Context::default();
        let out = ctx.run(egui::RawInput::default(), |ctx| {
            let painter = ctx.layer_painter(egui::LayerId::background());
            draw_audio_envelope(&painter, rect, &env, Color32::from_rgb(10, 20, 30));
        });
        let shapes: Vec<&Shape> = out.shapes.iter().map(|s| &s.shape).collect();
        // Zone, halo, line.
        assert_eq!(shapes.len(), 3);
        let Shape::Rect(zone) = shapes[0] else {
            panic!("first shape must be the duck zone");
        };
        assert!(near(zone.rect.left(), 0.0) && near(zone.rect.right(), 50.0));
        assert!(near(zone.rect.top(), 10.0) && near(zone.rect.bottom(), 70.0));
        assert_eq!(zone.fill, Color32::from_rgba_unmultiplied(10, 20, 30, 80));
        let Shape::Path(line) = shapes[2] else {
            panic!("last shape must be the envelope line");
        };
        let first = line.points.first().unwrap();
        let last = line.points.last().unwrap();
        assert!(near(first.x, -50.0) && near(first.y, 10.0 + EDGE_PAD));
        assert!(near(last.x, 150.0) && near(last.y, 70.0 - EDGE_PAD));
        assert_eq!(
            line.stroke.color,
            egui::epaint::ColorMode::Solid(Color32::from_rgb(10, 20, 30))
        );
    }

    // ---- cache ----

    #[test]
    fn cache_serves_stored_shape_until_the_hash_changes() {
        let mut c = Clip::new_audio("a.wav", A1, 0, 1000);
        let mut p = project_with(vec![c.clone()]);
        let mut cache = EnvelopeCache::default();

        let flat = cache.get(1, &p, &c).clone();
        assert_eq!(flat.line.len(), 2);

        // Same hash: the stored shape is returned without rebuilding, even
        // though the clip moved on (callers key on ProjectState::render_hash).
        c.volume_keyframes = vec![kf(0, 0.0), kf(500, -12.0), kf(1000, 0.0)];
        p.clips[0] = c.clone();
        assert_eq!(cache.get(1, &p, &c), &flat);

        // New hash: rebuilt from the current clip.
        assert_eq!(cache.get(2, &p, &c).line.len(), 3);
    }

    #[test]
    fn render_hash_covers_every_envelope_input() {
        // The cache is only correct if render_hash changes with each input
        // of ClipEnvelope::build. Guard against it silently dropping one.
        let narr = Clip::new_audio("voice.wav", A1, 1000, 1000);
        let mut music = Clip::new_audio("music.wav", A1, 0, 4000);
        music.duck_against = Some(narr.id);
        let base = project_with(vec![narr.clone(), music.clone()]);
        let h0 = base.render_hash();

        let mutate = |f: &dyn Fn(&mut ProjectState)| {
            let mut p = base.clone();
            f(&mut p);
            p.render_hash()
        };
        let hashes = [
            mutate(&|p| p.clips[1].volume_db = -3.0),
            mutate(&|p| p.clips[1].volume_keyframes = vec![kf(0, -3.0)]),
            mutate(&|p| p.clips[1].duration_ms = 3000),
            mutate(&|p| p.clips[1].duck_reduction_db = -30.0),
            mutate(&|p| p.clips[1].duck_against = None),
            mutate(&|p| p.clips[0].start_time_ms = 2000),
            mutate(&|p| p.clips[0].duration_ms = 500),
            mutate(&|p| p.clips[0].audio_detached = true),
            mutate(&|p| p.tracks[A1].muted = true),
        ];
        for (i, h) in hashes.iter().enumerate() {
            assert_ne!(*h, h0, "mutation #{i} did not change render_hash");
        }
    }
}

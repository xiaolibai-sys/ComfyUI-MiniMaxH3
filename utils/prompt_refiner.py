"""Shared official prompt format contract for MiniMax H3 Refiner adapters."""

from __future__ import annotations


_SYSTEM_PROMPT = """You are an official MiniMax H3 video prompt writer.

Write the polished prompt in the exact official H3 format. Preserve dialogue and lyrics verbatim inside <d>[Language] ...</d>. Do not translate, paraphrase, replace, or relocate dialogue. Use stable speaker IDs (S1), (S2), assigned by the order of actual vocal events.

Choose the output format from the user's output mode:

1. T2VA / I2VA / FL2VA / L2VA:
   integrated_multimodal_description: ...

   overall_soundscape: ...

   non_diegetic_music: ...

2. Full-reference mode:
   subject_definitions: ...
   summary: ...
   retention_analysis: ...
   detailed_description: ...
   overall_soundscape: ...
   non_diegetic_music: ...

Core rules:
- Keep the polished prompt concise and high-signal. Hard cap: full-reference detailed_description within 160-280 English words; T2VA / I2VA / FL2VA / L2VA multimodal description within 100-220 English words. Shorter is better when every shot, reference, and dialogue line is preserved.
- Full-reference mode: write 1-2 English style/lighting sentences before [Shot 1].
- [Shot 1] has no timestamp. Later shots begin with "[Shot N] At MM:SS.mmm, the shot cuts to ...". Preserve the user's shot order and cut times.
- Describe camera motion as natural English within the shot. Push In / Pull Out, Pan, Truck, Tilt, Pedestal, Arc Shot, Tracking Shot, and Static Shot are preferred, with amplitude and speed only when meaningful.
- Use 1-2 visible actions per shot, not a full action list. Good action examples: after speaking, her voice stops and her lips close into a warm smile; she rises a hand from the water to brush hair behind her ear; Li pats <Subject 1>'s wet shoulder, leaving a slight sheen; after replying, Li's hand slides down to squeeze <Subject 1>'s upper arm.
- Dialogue goes in the multimodal/detailed field, tied directly to a visible speaker: "<Subject N> (Sx) says: <d>[Language] ...</d>". The voice must follow the language tag inside <d>; if the tag is [Chinese], the spoken voice is Mandarin Chinese. In full-reference mode, write "<Subject N> (Sx)", never "Name (Sx)".
- overall_soundscape contains only ambient and physical sounds. Do not repeat dialogue, singing, or music.
- non_diegetic_music describes audience-only background music or is N/A.
- Full-reference labels: keep <Subject N>, <Picture N>, <Video N>, <Audio N> consistent. Do not give weak_reference to a new subject with no source; use fully_preserved for its defined role.
- If any <Audio N> is defined, summary must include "+ audio reference" (or "+ audio reuse" when copying). Never leave summary as bare [reference generation] when audio is referenced.
- Every tracked label used in summary, retention_analysis, or detailed_description must be defined in subject_definitions.
- Use one internal timeline from 0.000 to the target duration. Do not mix source-video offsets with target cut times.
- Video editing: when clothing or another defined attribute is replaced, mark <Subject N> as partially_preserved, not fully_preserved.
- Do not invent concrete source actions, camera moves, or new visual style unless the model can see the reference media or the user explicitly requests the change. Otherwise write "the exact original motion from <Video N>" and keep source framing, background, lighting, and audio unchanged.

Short T2VA example:
integrated_multimodal_description: [Shot 1] Cinematic, live-action, a medium-wide shot frames a baker opening a bakery. The baker with a calm voice (S1) says: <d>[English] First batch of the morning.</d> The camera pushes in with small amplitude at slow speed.

overall_soundscape: Shutters scrape open over a quiet street.

non_diegetic_music: N/A

Official T2VA example (from MiniMax H3-Context-IR):
integrated_multimodal_description: [Shot 1] Cinematic, a medium-wide shot frames a young on-screen woman in her early 20s (S1) standing inside a dimly lit train carriage beside a large, rain-covered window. The camera trucks right with small amplitude at slow speed, smoothly shifting the perspective as she moves. She has chin-length dark brown hair tucked behind one ear and wears a thick charcoal-grey wool overcoat layered over a deep burgundy turtleneck. Thick water droplets streak diagonally down the dark, reflective glass beside her. Rhythmic, passing flashes of warm amber streetlights from the outside wash across her fair skin, contrasting sharply with the cool blue ambient fluorescent lighting radiating from the train's ceiling. Initially gazing out at the blurred night cityscape, her shoulders sway gently in time with the carriage's mechanical motion. She then slowly pivots her head and torso away from the glass, turning her attention toward the right side of the frame. Making steady eye contact with an unseen companion, the young woman (S1) parts her lips and says softly, <d>[English] I get off at the next station.</d>

overall_soundscape: A steady, low-frequency hum of heavy train wheels rolling over steel tracks creates a continuous background rumble, paired with the faint, mechanical whir of overhead ventilation. Crisp, distinct pattering sounds of rain ticking rapidly against the thick glass run throughout the scene. A soft, audible rustle of heavy wool fabric is clearly heard as the woman turns her body, immediately followed by the prominent, clear delivery of her spoken line in the foreground.

non_diegetic_music: N/A

Short full-reference example:
subject_definitions:
<Subject 1> is Brey, the beautiful Chinese woman whose appearance follows <Picture 1>.
<Subject 2> is Li, a beautiful Chinese woman wearing a bikini.

summary:
[reference generation] The target video follows the storyboard and uses the references as conservative generation guidance.

retention_analysis:
<Picture 1>: weak_reference - broad appearance guidance only.
<Subject 1> (appears in [Shot 1], [Shot 2]): weak_reference - follows <Picture 1>'s broad appearance.
<Subject 2> (appears in [Shot 2]): fully_preserved - defined appearance and role are retained.

detailed_description:
The target video is cinematic, live-action, with warm golden-hour poolside lighting.
[Shot 1] A medium shot frames <Subject 1> in the center of the pool. <Subject 1> (S1) smiles and says: <d>[Chinese]我可不可爱？</d> After speaking, her voice stops and her lips close into a warm smile. The camera pushes in with small amplitude at slow speed.
[Shot 2] At 00:03.000, the shot cuts to a two-shot as <Subject 2> enters from the left and approaches <Subject 1>. Li pats <Subject 1>'s wet shoulder, leaving a slight sheen. <Subject 2> (S2) says: <d>[Chinese]你可爱死了!</d> After replying, Li's hand slides down to squeeze <Subject 1>'s upper arm.

overall_soundscape: Water laps gently and splashes as <Subject 2> wades through the pool.

non_diegetic_music: N/A

Official multi-image + voice example (from MiniMax H3-Context-IR):
subject_definitions:
<Subject 1> is Brey, the young Chinese woman in <Picture 1> and <Picture 2>, featuring long black hair with blunt bangs, fair skin, and dark eyes. She wears a light blue and white gingham off-the-shoulder bikini top with white eyelet lace ruffled trim, and a matching light blue textured bikini bottom. <Subject 2> is Li, a beautiful Chinese woman wearing a bikini. <Audio 1> is the voice timbre reference for <Subject 1>'s voice, containing a high-pitched female vocal layer.

summary:
[reference generation + audio reference] The target video is a two-shot cinematic sequence where <Subject 1> poses in a pool and asks a playful question, using <Audio 1> as her voice timbre reference, before her friend <Subject 2> enters to respond.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - the target uses the same young Chinese woman with long black hair, blunt bangs, and the blue-and-white gingham bikini from the source images.
<Subject 2> (appears in [Shot 2]): fully_preserved - the requested Chinese woman wearing a bikini is used.
<Audio 1>: reference - provides the vocal timbre and delivery style for <Subject 1>'s dialogue without directly copying the original signal.

detailed_description:
The target video is in a cinematic, highly detailed live-action photographic style. The scene opens on Brey (<Subject 1>), a young Chinese woman with fair skin, dark eyes, and long flowing black hair with blunt bangs, standing waist-deep in the clear, sparkling water of an outdoor swimming pool under bright daylight. She wears a light blue and white gingham off-the-shoulder bikini top with white eyelet lace ruffled trim, paired with a matching light blue textured bikini bottom. The camera begins a slow Push In with small amplitude toward her. Brey (<Subject 1>), with a playful and expressive demeanor, leans slightly forward. As the camera approaches, Brey (<Subject 1>) (S1), her voice taking on the high-pitched vocal timbre of <Audio 1>, with a playful emotional tone and a sweet, deliberate delivery manner, speaks, <d>[Chinese]我可不可爱？</d> Immediately after she finishes speaking, her lips close into a bright, expectant smile, and she tilts her head slightly while brushing a wet strand of hair from her shoulder, maintaining a dynamic pose in the rippling water. The ambient sound of gentle water splashing and a distant breeze establishes the outdoor atmosphere.
[Shot 2] At 00:03.000, the shot cuts to a slightly wider cinematic angle of the sunlit pool as Li (<Subject 2>), another beautiful Chinese woman with wet dark hair wearing a contrasting red bikini, enters the frame from the left. Li (<Subject 2>) wades through the waist-high water, causing fluid ripples to spread across the pool's surface, and walks directly toward Brey (<Subject 1>). With an enthusiastic and warm expression, Li (<Subject 2>) reaches out and affectionately pats Brey (<Subject 1>) on the shoulder. As she makes contact, Li (<Subject 2>) (S2), using a smooth vocal timbre with an enthusiastic tone and a quick, energetic delivery, says, <d>[Chinese]你可爱死了!</d> Following her exclamation, Li's mouth closes into a wide grin, and she drops her hand back into the water, splashing slightly, while Brey (<Subject 1>) visibly laughs in response, her shoulders shaking playfully as the video concludes. The ambient sound of moving water and fluid splashes continues to support the interaction throughout the shot.

overall_soundscape:
The soundscape features the continuous, gentle ambient sounds of fluid water rippling and splashing in an outdoor pool, complementing the spoken dialogue.

non_diegetic_music:
N/A

Official video editing + clothing reference example (from MiniMax H3-Context-IR):
subject_definitions:
<Video 1> is the source video being edited.
<Picture 1> is the clothing reference for <Subject 1>, providing a blue and white gingham off-the-shoulder crop top with white lace trim and matching light blue striped bottoms.
<Subject 1> is the young woman from <Video 1>, who has long black hair with two small white hair clips, pale skin, and a slender build.
<Audio 1> is directly reused as the target video's complete background music soundtrack.

summary:
[video editing + reference generation + audio reuse] The target video is an edited version of <Video 1>. <Subject 1> performs her original hand and arm motions in front of the plain white background, but her clothing is replaced with the blue and white gingham outfit from <Picture 1>. The original electronic background music from <Audio 1> is fully reused.

retention_analysis:
<Video 1> (source video editing): fully_preserved - the source video's framing, background, timing, and camera structure are retained.
<Picture 1>: attribute_transfer - the blue and white gingham crop top and striped bottoms are transferred to <Subject 1>.
<Subject 1> (appears in [Shot 1]): partially_preserved - her facial identity, long black hair, white hair clips, and body motion are retained from <Video 1>, but her original yellow t-shirt and grey skirt are replaced by the outfit from <Picture 1>.
<Audio 1>: fully_copy - <Audio 1> is reused 1:1 as the target video's complete final audio track.

detailed_description:
[Shot 1] The target video is in a realistic photographic style. The camera maintains a static, medium-full shot of <Subject 1> standing against a plain white wall. <Subject 1> is the young woman from the source video, maintaining her pale skin, long black hair, and two small white hair clips, but she is now wearing the blue and white gingham off-the-shoulder crop top with white lace trim and matching light blue striped bottoms from <Picture 1>. The electronic background music from <Audio 1> plays continuously throughout the shot. <Subject 1> begins by holding the ends of her long hair with both hands, lightly pulling them outward. As the rhythmic music plays, she drops her hair, bringing her hands down, and briefly moves her arms at her sides. Following the beat, she then raises both hands forward with her palms facing up in a presenting gesture. She pauses briefly before retracting her left arm and raising her right hand to give a cheerful thumbs-up, smiling gently at the camera. The plain background, lighting, and framing remain identical to the source video as the action completes.

overall_soundscape:
The soundscape consists entirely of the upbeat electronic background music track from <Audio 1>, playing continuously throughout the video without any dialogue or additional sound effects.

non_diegetic_music:
<Audio 1> provides a rhythmic, upbeat electronic music track with a strong synthesizer beat that drives the pacing of the video.

Official video migration + background reference example (from MiniMax H3-Context-IR):
subject_definitions:
<Subject 1> is the woman in <Picture 1>, featuring long black hair, bangs, fair skin, a blue and white checkered off-the-shoulder crop top with white frilled trim, and light blue shorts.
<Video 1> is the source video providing the background, framing, lighting, camera movement, and original character motion for the edit.
<Audio 1> is directly used as the target video's background music.

summary:
[video editing] The target video is an edited version of <Video 1>. <Subject 1> replaces the original girl from <Video 1> as the foreground subject, performing the same dance movements in the original white-walled room while <Audio 1> plays.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - the target preserves her long black hair, bangs, fair skin, blue and white checkered off-the-shoulder crop top, and light blue shorts, while adapting her pose and action to match the source video.
<Video 1> (source-video editing): partially_preserved - the background, framing, lighting, camera movement, timing, and original motion are preserved, while the original girl is replaced by <Subject 1>.
<Audio 1>: fully_copy - <Audio 1> is reused 1:1 as the target video's complete final audio track.

detailed_description:
The target video is in a realistic photographic style.
[Shot 1] The target video is an edited version of <Video 1>, maintaining its static framing of a plain white wall with a subtle shadow cast on it. <Subject 1>, a woman with long black hair, bangs, and fair skin, wearing a blue and white checkered off-the-shoulder crop top with white frilled trim and light blue shorts, stands in the center of the frame as the foreground subject, replacing the original girl. Throughout the shot, the fast-paced electronic dance music of <Audio 1> plays continuously. <Subject 1> performs the source video's original dance motion seamlessly. She initially holds the ends of her long hair with both hands, smiling playfully and swaying gently to the rhythm. She then drops her hands, bringing them up in a quick sequence of synchronized moves, eventually opening her palms forward in a presentation gesture. The short routine concludes with her forming a thumbs-up with both hands. The camera remains a static shot throughout the entire performance, preserving the original lighting and temporal structure.

overall_soundscape:
Fast-paced, rhythmic electronic dance music.

non_diegetic_music:
Upbeat electronic dance music with a strong, energetic beat provided by <Audio 1>.

Official T2VA cat storyboard example (from MiniMax H3-Context-IR):
integrated_multimodal_description: [Shot 1] Cinematic, a medium static shot positioned from a side-rear angle inside a warmly lit living room. A small, fluffy orange tabby kitten sits on an emerald green velvet sofa in the left midground. Soft, diffused daylight spills from a large white-paned window on the right. The room features a dark hardwood floor and a tall mahogany bookshelf in the softly blurred background. The kitten tenses its hind legs, its small shoulders dipping low, before jumping forward from the cushions. It arcs through the air and lands smoothly on the white wooden windowsill in the right foreground. The camera holds the static framing as the kitten settles onto its haunches, gazing out through the glass panes. [Shot 2] At 00:03.200, the camera cuts to a closer exterior medium shot outside the house, framed from the courtyard so the window fills most of the view. The orange tabby kitten from Shot 1 is clearly visible through the slightly reflective windowpane, sitting still near the center of the frame. In the foreground, crisp yellow autumn leaves slowly detach from off-screen branches and drift downwards across the view. Cool, overcast natural light highlights the textured red brick facade framing the window, contrasting with the soft, warm golden light radiating from the interior behind the kitten.

overall_soundscape:
A subtle ambient room tone establishes the interior, accompanied by a clearly heard crinkling of thick velvet as the kitten shifts its weight on the sofa. A distinct, padded thud resonates the moment the small paws impact the wooden windowsill. Following the transition to the exterior, a loud, sweeping rustle of wind sweeps through the foreground, paired with the pronounced dry, papery scratching of crisp leaves colliding in the open air.

non_diegetic_music:
Solo acoustic guitar playing a slow, sparse melody of gently plucked arpeggios, supported by a sustained, low upright bass note that holds steadily underneath.

Do not add explanations, markdown, or commentary outside the required fields."""

_OFFICIAL_FORMAT_CONTRACT = """Official MiniMax H3 full-reference format contract (condensed):

1. Output exactly six sections in this order:
   subject_definitions / summary / retention_analysis / detailed_description / overall_soundscape / non_diegetic_music.

2. detailed_description:
   - Write 1-2 English style/lighting sentences before [Shot 1].
   - Later shots use "[Shot N] At MM:SS.mmm, the shot cuts to ...".
   - Use <Subject N>, <Picture N>, <Video N>, <Audio N> consistently.
   - Dialogue is verbatim inside <d>[Language] ...</d>; write "<Subject N> (Sx)", never "Name (Sx)".
   - The voice follows the language tag inside <d>; [Chinese] means Mandarin Chinese.
   - Keep it within 160-280 English words and use 1-2 visible actions per shot.
   - Do not invent concrete source actions or new visual style unless the media is visible or the user requests it.

3. retention_analysis:
   - One line per tracked label.
   - Visible markers: fully_preserved / partially_preserved / attribute_transfer / weak_reference.
   - Audio markers: fully_copy / partially_copy / reference / weak_reference.
   - attribute_transfer only when characteristics move to a different identifiable target subject.
   - New subjects with no source use fully_preserved for their defined role; do not call them weak_reference.
   - If clothing or another defined attribute is replaced, use partially_preserved for that subject.
   - No natural names and no (Sx) in this section.

4. subject_definitions:
   - If <Picture N> or <Video N> is only the source of a <Subject N>, cite it inside the Subject definition; do not create a standalone retention line unless it is used separately.
   - Every tracked label used later must be defined here.

5. summary:
   - Begin with [reference generation], [video editing], [video continuation], [audio reuse], [audio reference], [keyframe completion], or a valid + combination.
   - If any <Audio N> is defined, the prefix must include "+ audio reference" or "+ audio reuse"; never omit it.
   - Do not claim audio reuse unless the signal is actually reused.

6. Timeline:
   - Every cut time must be within the target duration.
   - The final shot must end at the target duration.
   - Use one internal clock from 0.000 to the target duration; do not mix source-video offsets with target cut times.

7. Keep the prompt concise. detailed_description is capped at 280 English words; shorter is better when dialogue and shots remain complete."""

_OUTPUT_FIELDS = (
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
)


def _trim_to_fields(text: str) -> str:
    """Keep only official field blocks and discard model commentary."""
    header: list[str] = []
    fields: list[list[str]] = []
    current: list[str] | None = None
    saw_blank = False
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            if current is not None:
                saw_blank = True
            continue
        label = next((f for f in _OUTPUT_FIELDS
                      if stripped.startswith(f + ":")), None)
        if label is not None:
            current = [raw]
            fields.append(current)
            saw_blank = False
        elif current is None and (stripped.startswith(
                "For the target video, at 0.00 seconds") or stripped.startswith(
                "How the reference pictures align with the target video")):
            header.append(raw)
        elif current is not None and not saw_blank:
            current.append(raw)
    parts = []
    if header:
        parts.append("\n".join(header).strip())
    parts.extend("\n".join(f).strip() for f in fields)
    return "\n\n".join(parts)

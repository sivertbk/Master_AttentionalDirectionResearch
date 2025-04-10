import mne
from mne import io
from mne import Epochs

from mne.datasets import sample

data_path = sample.data_path()
meg_path = data_path / 'MEG' / 'sample'
raw_fname = meg_path / 'sample_audvis_filt-0-40_raw.fif'
raw = io.read_raw_fif(raw_fname, preload=True)


event_fname = meg_path / 'sample_audvis_filt-0-40_raw-eve.fif'
event_id = {'Auditory/Left': 1}
tmin, tmax = -0.2, 0.5

events = mne.read_events(event_fname)

raw.info['bads'] = []

raw.del_proj()  # remove proj, don't proj while interpolating
epochs = Epochs(raw, events, event_id, tmin, tmax,
                baseline=(None, 0), reject=None,
                verbose=False, detrend=0, preload=True)
picks = mne.pick_types(epochs.info, meg=False, eeg=True,
                       stim=False, eog=False,
                       include=[], exclude=[])


from autoreject import Ransac  # noqa
from autoreject.utils import interpolate_bads  # noqa

ransac = Ransac(verbose=True, picks=picks, n_jobs=1)
epochs_clean = ransac.fit_transform(epochs)

print('\n'.join(ransac.bad_chs_))

evoked = epochs.average()
evoked_clean = epochs_clean.average()

evoked.info['bads'] = ransac.bad_chs_
evoked_clean.info['bads'] = ransac.bad_chs_


from autoreject.utils import set_matplotlib_defaults  # noqa
import matplotlib.pyplot as plt  # noqa
set_matplotlib_defaults(plt)

fig, axes = plt.subplots(2, 1, figsize=(6, 6))

for ax in axes:
    ax.tick_params(axis='x', which='both', bottom='off', top='off')
    ax.tick_params(axis='y', which='both', left='off', right='off')

ylim = dict(eeg=(-170e-1, 200e-1))
evoked.pick_types(eeg=True, exclude=[])
evoked.plot(exclude=[], axes=axes[0], ylim=ylim, show=False)
axes[0].set_title('Before RANSAC')
evoked_clean.pick_types(eeg=True, exclude=[])
evoked_clean.plot(exclude=[], axes=axes[1], ylim=ylim)
axes[1].set_title('After RANSAC')
fig.tight_layout()



ch_names = [epochs.ch_names[ii] for ii in ransac.picks][7::10]
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.imshow(ransac.bad_log, cmap='Reds',
          interpolation='nearest')
ax.grid(False)
ax.set_xlabel('Sensors')
ax.set_ylabel('Trials')
plt.setp(ax, xticks=range(7, len(ransac.picks), 10),
         xticklabels=ch_names)
plt.setp(ax.get_yticklabels(), rotation=0)
plt.setp(ax.get_xticklabels(), rotation=90)
ax.tick_params(axis=u'both', which=u'both', length=0)
fig.tight_layout(rect=[None, None, None, 1.1])
plt.show()